"""End-to-end prediction pipeline.

Trains the per-track ensemble on the full history available, generates
predictions for every (rm_id, end_date) in ``prediction_mapping``, and
writes a submission CSV. Includes sanity checks: row count, monotonicity,
non-negativity, ID alignment.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from src.data import (
    DATA_DIR,
    REPO_ROOT,
    Datasets,
    build_profile,
    cumulative_truth,
    load_or_build,
)
from src.ensemble import EnsembleConfig, blend, historical_cap_table
from src.features import build_features
from src.gating import assign_tracks
from src.models.empirical import EmpiricalQuantileForecaster
from src.models.lgbm_quantile import LGBMQuantileForecaster, assemble_training_set
from src.models.linear_per_rm import PerRMLinearForecaster
from src.anchor import AnchorConfig, build_anchor, combine_with_slope
from src.regime import classify_regime, shrink_per_rm
from src.seasonality import SeasonalityConfig, build_shape_lookup, compute_shapes
from src.validation import DEFAULT_FOLDS, build_query_for_fold, evaluate

SUBMISSIONS_DIR = REPO_ROOT / "submissions"
SUBMISSIONS_DIR.mkdir(parents=True, exist_ok=True)

DEFAULT_ENSEMBLE = EnsembleConfig(
    track_weights={
        "A": {"linear": 1.0},
        "B": {"linear": 1.0},
        "C": {"linear": 1.0},  # linear with low per-rm shrink (0.3); see DEFAULT_PER_TRACK_SHRINK
        "D": {},
    },
    conservative_shrink=1.0,
    cap_multiplier=None,
    enforce_monotone=True,
    floor_zero=True,
)

# Per-track slope shrinks, tuned on the joint walk-forward CV.
# v8 uses higher shrinks because the q=0.30 pairwise-slope estimator is
# more conservative than Theil-Sen — the slope itself is smaller, so the
# applied shrink can be larger without over-predicting.
DEFAULT_PER_TRACK_SHRINK = {"A": 0.70, "B": 0.70, "C": 0.30}    # v3
V7_PER_TRACK_SHRINK = {"A": 0.70, "B": 0.70, "C": 0.40}           # v7
V8_PER_TRACK_SHRINK = {"A": 0.80, "B": 0.80, "C": 0.50}           # v8

DEFAULT_TRAILING_WINDOW_DAYS = 210
V8_PAIR_QUANTILE = 0.30


@dataclass
class PredictionRun:
    submission: pd.DataFrame  # ID, predicted_weight
    detail: pd.DataFrame  # rm_id, forecast_end_date, predicted_weight, track
    tracks: pd.DataFrame
    cv_scores: dict


def train_models_and_predict(
    ds: Datasets,
    history_end: pd.Timestamp,
    target_year: int,
    end_dates: pd.DatetimeIndex,
    rm_ids: list[int],
    model: str = "v8",
    per_track_shrink: dict[str, float] | None = None,
    use_regime: bool = False,
    use_anchor: bool = False,
    anchor_alpha: float = 0.65,
    use_lgbm: bool = False,
    use_nhits: bool = False,
    nhits_max_epochs: int = 30,
    track_weights: dict | None = None,
    conservative_shrink: float = 1.0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Train the ensemble and predict for one cutoff/target combination.

    ``model`` selects the slope estimator and the within-window shape:
        - ``"v3"``: OLS slope on a 210-day trailing window, uniform shape.
          Per-track shrink A=B=0.70, C=0.30.
        - ``"v7"``: Theil-Sen slope (q=0.5 pairwise) + empirical seasonal shape.
          Per-track shrink A=B=0.70, C=0.40.
        - ``"v8"`` (default): Lower-quantile pairwise slope (q=0.30) + same
          empirical seasonal shape. Per-track shrink A=B=0.80, C=0.50.
          The q=0.30 slope is a more conservative estimator naturally aligned
          with τ=0.2 pinball loss, allowing larger shrink without over-prediction.

    Both models share the same gating and Track-D-zero behaviour.
    Walk-forward CV (avg pinball, lower better): v3 9693, v7 9183, v8 8944.
    """
    daily_pre = ds.daily[ds.daily["date"] < history_end]
    profile = build_profile(daily_pre, year=target_year - 1)
    tracks = assign_tracks(profile, all_rm_ids=rm_ids)
    track_a_ids = set(tracks[tracks["track"] == "A"]["rm_id"].tolist())
    track_b_ids = set(tracks[tracks["track"] == "B"]["rm_id"].tolist())
    track_c_ids = set(tracks[tracks["track"] == "C"]["rm_id"].tolist())

    query = (
        pd.MultiIndex.from_product(
            [rm_ids, end_dates], names=["rm_id", "forecast_end_date"]
        )
        .to_frame(index=False)
    )

    # Model A — empirical
    emp = EmpiricalQuantileForecaster(tau=0.2, min_year=2020).fit(ds.daily, history_end=history_end)
    preds_emp = emp.predict(query)

    # Default per-track shrink — picks v3/v7/v8 defaults if not overridden.
    if per_track_shrink is not None:
        shrinks = per_track_shrink
    elif model == "v8":
        shrinks = V8_PER_TRACK_SHRINK
    elif model == "v7":
        shrinks = V7_PER_TRACK_SHRINK
    else:
        shrinks = DEFAULT_PER_TRACK_SHRINK
    per_rm_shrink = {
        **{rm: shrinks.get("A", 0.7) for rm in track_a_ids},
        **{rm: shrinks.get("B", 0.7) for rm in track_b_ids},
        **{rm: shrinks.get("C", 0.3) for rm in track_c_ids},
    }
    if use_regime:
        regimes = classify_regime(ds.daily, cutoff=history_end)
        intermittent_ids = set(
            regimes[regimes["regime"] == "INTERMITTENT"]["rm_id"].astype(int).tolist()
        )
        for rm in list(per_rm_shrink.keys()):
            if rm in intermittent_ids:
                del per_rm_shrink[rm]
    forecast_active = set(per_rm_shrink.keys())

    # Slope estimator: OLS for v3, Theil-Sen (q=0.5 pairwise) for v7,
    # lower-quantile pairwise (q=0.30) for v8.
    if model == "v8":
        slope_strategy = "trailing_window_qpairs"
    elif model == "v7":
        slope_strategy = "trailing_window_theilsen"
    else:
        slope_strategy = "trailing_window"
    lin = PerRMLinearForecaster(
        fit_year=target_year - 1,
        slope_strategy=slope_strategy,
        trailing_window_days=DEFAULT_TRAILING_WINDOW_DAYS,
        cutoff=history_end,
        slope_shrink=1.0,
        pair_quantile=V8_PAIR_QUANTILE,
    ).fit(ds.daily)

    if model in ("v7", "v8"):
        # v7: empirical seasonal shape replaces uniform d/151. The May-31
        # prediction is identical (slope × 151 × shrink); only intermediate
        # end_dates change. Both folds improve when this is enabled.
        per_rm_shape, global_shape, n_active_per_rm = compute_shapes(
            ds.daily[ds.daily["date"] < history_end],
            fit_year=target_year - 1,
            cfg=SeasonalityConfig(),
        )
        shape_lookup = build_shape_lookup(
            forecast_active, per_rm_shape, global_shape, n_active_per_rm, SeasonalityConfig()
        )
        # Build predictions manually: total_at_151 × shape(d).
        slope_arr = np.array([
            (lin.fits.get(int(rm), (0.0, False))[0]
             if lin.fits.get(int(rm), (0.0, False))[1]
             else 0.0)
            for rm in query["rm_id"]
        ])
        shrink_arr = np.array([per_rm_shrink.get(int(rm), 0.0) for rm in query["rm_id"]])
        active_mask = np.array([int(rm) in forecast_active for rm in query["rm_id"]])
        total_pred = slope_arr * 151.0 * shrink_arr * active_mask
        doys = query["forecast_end_date"].dt.dayofyear.to_numpy()
        # Build shape value per row.
        shape_vals = np.array([
            shape_lookup.get(int(rm), global_shape)[int(d) - 1] if 1 <= int(d) <= 151 else 1.0
            for rm, d in zip(query["rm_id"].to_numpy(), doys)
        ])
        preds_lin = query[["rm_id", "forecast_end_date"]].copy()
        preds_lin["predicted_weight"] = np.maximum(0.0, total_pred * shape_vals)
    else:
        preds_lin = lin.predict(
            query,
            rm_id_track_filter=forecast_active,
            per_rm_shrink=per_rm_shrink,
        )

    model_preds: dict[str, pd.DataFrame] = {"empirical": preds_emp, "linear": preds_lin}

    if use_nhits and track_a_ids:
        from src.models.nhits_quantile import NHITSQuantileForecaster

        nh = NHITSQuantileForecaster(
            rm_ids_to_train=sorted(track_a_ids),
            horizon=151,
            input_size=730,
            max_epochs=nhits_max_epochs,
            hidden_size=128,
            batch_size=64,
        )
        nh.fit(ds.daily, history_end=history_end)
        preds_nh = nh.predict_cumulative(target_year=target_year, end_dates=pd.DatetimeIndex(sorted(end_dates)))
        model_preds["nhits"] = preds_nh

    # Model B — LightGBM (optional — costly + not always helpful)
    if use_lgbm:
        train_years = [y for y in range(2020, target_year) if y < target_year]
        base_end = pd.date_range("2020-01-02", "2020-05-31", freq="D")
        X, y, w = assemble_training_set(
            daily=daily_pre,
            materials=ds.materials,
            train_years=train_years,
            end_dates=base_end,
            rm_ids=rm_ids,
            profile_for_weight=profile,
        )
        infer_out = build_features(daily_pre, ds.materials, target_year, end_dates, rm_ids)
        # Use the last train year as a tiny holdout for early stopping.
        last_train = max(train_years)
        last_train_dates = pd.date_range(f"{last_train}-01-02", f"{last_train}-05-31", freq="D")
        valid_out = build_features(daily_pre, ds.materials, last_train, last_train_dates, rm_ids)
        valid_truth = build_features(ds.daily, ds.materials, last_train, last_train_dates, rm_ids).target

        m_lgbm = LGBMQuantileForecaster()
        m_lgbm.fit(X, y, valid_df=valid_out.features, valid_target=valid_truth, sample_weight=w)
        preds_lgbm = m_lgbm.predict(infer_out.features)
        model_preds["lgbm"] = preds_lgbm

    cap = historical_cap_table(daily_pre, target_year - 1)
    cfg = EnsembleConfig(
        track_weights=track_weights or DEFAULT_ENSEMBLE.track_weights,
        conservative_shrink=conservative_shrink,
        cap_multiplier=None,
        enforce_monotone=True,
        floor_zero=True,
    )
    blended = blend(model_preds, tracks, cfg, historical_cap=cap)

    # Production-only: lift predictions toward last year's H1 trajectory
    # via max(slope_pred, alpha * prior_year_jan_may_at_doy). This is a
    # calculated bet — see src/anchor.py for the rationale and risks.
    if use_anchor:
        anchor_preds = build_anchor(
            ds.daily, target_year, end_dates, AnchorConfig(alpha=anchor_alpha)
        )
        # Restrict the anchor to rm_ids that the slope path is also forecasting
        # so we don't accidentally introduce predictions for INTERMITTENT or
        # Track-D rm_ids (those should remain zero).
        anchor_preds = anchor_preds[anchor_preds["rm_id"].isin(forecast_active)]
        blended = combine_with_slope(blended, anchor_preds)
        # Re-enforce monotonicity after the anchor lift.
        blended = blended.sort_values(["rm_id", "forecast_end_date"]).reset_index(drop=True)
        blended["predicted_weight"] = (
            blended.groupby("rm_id")["predicted_weight"].cummax().to_numpy()
        )

    blended = blended.merge(tracks, on="rm_id", how="left")
    return blended, tracks


def cv_score(
    ds: Datasets,
    rm_ids: list[int],
    model: str = "v8",
    per_track_shrink: dict[str, float] | None = None,
    use_regime: bool = False,
    use_lgbm: bool = False,
    use_nhits: bool = False,
    nhits_max_epochs: int = 30,
    track_weights: dict | None = None,
    conservative_shrink: float = 1.0,
) -> dict:
    """CV scoring — anchor is intentionally never enabled here."""
    out: dict[str, dict] = {}
    for fold in DEFAULT_FOLDS:
        end_dates = pd.date_range(
            f"{fold.target_year}-01-02", f"{fold.target_year}-05-31", freq="D"
        )
        blended, tracks = train_models_and_predict(
            ds=ds,
            history_end=fold.train_end + pd.Timedelta(days=1),
            target_year=fold.target_year,
            end_dates=end_dates,
            rm_ids=rm_ids,
            model=model,
            per_track_shrink=per_track_shrink,
            use_regime=use_regime,
            use_anchor=False,
            use_lgbm=use_lgbm,
            use_nhits=use_nhits,
            nhits_max_epochs=nhits_max_epochs,
            track_weights=track_weights,
            conservative_shrink=conservative_shrink,
        )
        out[fold.name] = evaluate(
            blended[["rm_id", "forecast_end_date", "predicted_weight"]], fold, ds.daily
        )
    return out


def make_submission(
    ds: Datasets,
    model: str = "v8",
    per_track_shrink: dict[str, float] | None = None,
    use_regime: bool = False,
    use_anchor: bool = False,
    anchor_alpha: float = 0.65,
    manual_overrides: dict[int, callable] | None = None,
    use_lgbm: bool = False,
    use_nhits: bool = False,
    nhits_max_epochs: int = 30,
    track_weights: dict | None = None,
    conservative_shrink: float = 1.0,
    label: str | None = None,
) -> PredictionRun:
    rm_ids = sorted(ds.daily["rm_id"].unique().tolist())

    pm = ds.prediction_mapping
    end_dates = pd.DatetimeIndex(sorted(pm["forecast_end_date"].unique()))

    detail, tracks = train_models_and_predict(
        ds=ds,
        history_end=pd.Timestamp("2025-01-01"),
        target_year=2025,
        end_dates=end_dates,
        rm_ids=rm_ids,
        model=model,
        per_track_shrink=per_track_shrink,
        use_regime=use_regime,
        use_anchor=use_anchor,
        anchor_alpha=anchor_alpha,
        use_lgbm=use_lgbm,
        use_nhits=use_nhits,
        nhits_max_epochs=nhits_max_epochs,
        track_weights=track_weights,
        conservative_shrink=conservative_shrink,
    )

    # Apply documented manual overrides per Phase 5 of the plan.
    if manual_overrides:
        for rm_id, override_fn in manual_overrides.items():
            mask = detail["rm_id"] == rm_id
            if not mask.any():
                continue
            sub = detail.loc[mask, ["rm_id", "forecast_end_date", "predicted_weight"]].copy()
            new_vals = override_fn(sub)  # callable returns a Series of new predicted_weight, indexed like sub
            detail.loc[mask, "predicted_weight"] = np.asarray(new_vals).clip(min=0.0)

    submission = pm.merge(
        detail[["rm_id", "forecast_end_date", "predicted_weight"]],
        on=["rm_id", "forecast_end_date"],
        how="left",
    )
    submission["predicted_weight"] = submission["predicted_weight"].fillna(0.0).clip(lower=0.0)
    submission = submission[["ID", "predicted_weight"]].sort_values("ID").reset_index(drop=True)

    cv_scores = cv_score(
        ds=ds,
        rm_ids=rm_ids,
        model=model,
        per_track_shrink=per_track_shrink,
        use_regime=use_regime,
        use_lgbm=use_lgbm,
        use_nhits=use_nhits,
        nhits_max_epochs=nhits_max_epochs,
        track_weights=track_weights,
        conservative_shrink=conservative_shrink,
    )

    sanity_checks(submission, detail, ds.prediction_mapping)

    if label:
        ts = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        path = SUBMISSIONS_DIR / f"submission_{label}_{ts}.csv"
        submission.to_csv(path, index=False)
        print(f"wrote {path}")

    return PredictionRun(
        submission=submission, detail=detail, tracks=tracks, cv_scores=cv_scores
    )


def sanity_checks(submission: pd.DataFrame, detail: pd.DataFrame, prediction_mapping: pd.DataFrame) -> None:
    """Raise loudly on any submission-format violation."""
    assert len(submission) == len(prediction_mapping), (
        f"row count mismatch: {len(submission)} vs {len(prediction_mapping)}"
    )
    assert set(submission["ID"]) == set(prediction_mapping["ID"]), "ID set mismatch"
    assert (submission["predicted_weight"] >= 0).all(), "negative predictions present"

    # Per-rm monotonicity check on detail.
    issues = (
        detail.sort_values(["rm_id", "forecast_end_date"])
        .groupby("rm_id")["predicted_weight"]
        .apply(lambda s: bool((s.diff().dropna() < -1e-6).any()))
    )
    bad = issues[issues]
    if not bad.empty:
        raise AssertionError(f"non-monotonic predictions for {len(bad)} rm_ids: {bad.index.tolist()[:5]}")
    print(f"sanity OK: {len(submission)} rows, all non-negative, monotonic per rm_id")


def main() -> None:
    """Build the production submission.

    Default model is v8: lower-quantile pairwise slope (q=0.30) + empirical
    seasonal shape, with per-track shrink A=B=0.80, C=0.50.

    Walk-forward CV: avg pinball 8944 (v7 9183, v3 9693), p2023=9041,
    p2024=8847. Both folds improve over v7 by ≥1%, fold gap 194 (v7 286,
    v3 843).

    v7 and v3 remain callable via ``model="v7"`` / ``model="v3"``.
    """
    ds = load_or_build()
    run = make_submission(ds=ds, model="v8", label="v8_qpairs_seasonal")
    print("\nCV scores (lower is better):")
    for name, s in run.cv_scores.items():
        print(f"  {name}: pinball={s['mean_pinball']:.1f}")
    avg = sum(s["mean_pinball"] for s in run.cv_scores.values()) / len(run.cv_scores)
    gap = abs(
        run.cv_scores["pretend-2023"]["mean_pinball"]
        - run.cv_scores["pretend-2024"]["mean_pinball"]
    )
    print(f"  average: pinball={avg:.1f}  gap={gap:.0f}")
    print("\nTrack distribution:")
    print(run.tracks["track"].value_counts().sort_index())


if __name__ == "__main__":
    main()
