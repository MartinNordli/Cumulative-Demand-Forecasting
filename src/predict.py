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

# Per-track slope shrink, tuned on the joint walk-forward CV across
# pretend-2023 and pretend-2024 with a 210-day trailing-window slope fit:
# - A: 0.70 (high R^2, stable cohort)
# - B: 0.70 (still active, more variable)
# - C: 0.30 (sparse but had some H2 delivery; the small contribution helps
#   recover loss on rm_ids that ramped up late and would otherwise score 0)
DEFAULT_PER_TRACK_SHRINK = {"A": 0.70, "B": 0.70, "C": 0.30}
# 210-day trailing window captures recent slope better than the full prior
# year — critical for rm_ids that started or ramped up mid-year.
DEFAULT_TRAILING_WINDOW_DAYS = 210


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
    per_track_shrink: dict[str, float] | None = None,
    use_regime: bool = True,
    use_anchor: bool = False,
    anchor_alpha: float = 0.65,
    use_lgbm: bool = False,
    use_nhits: bool = False,
    nhits_max_epochs: int = 30,
    track_weights: dict | None = None,
    conservative_shrink: float = 1.0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Train the ensemble and predict for one cutoff/target combination.

    When ``use_regime=True`` (default), per-rm shrink comes from the YoY
    trend regime classifier (``src/regime.py``); otherwise the legacy
    Track A/B/C per-track shrink is used.

    When ``use_anchor=True`` (production-only — never CV), the slope-based
    prediction is combined with a prior-year-Jan-May trajectory anchor via
    ``max(slope_pred, alpha * prior_year_jan_may[doy])``.
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

    # Per-rm shrink: per-track baseline (v3) + INTERMITTENT override.
    # The full regime model (GROWING/STABLE/...) over-fits one fold and
    # regresses the other. The INTERMITTENT class — rm_ids silent for 60+
    # days with minimal recent activity — generalises across folds and
    # consistently catches false-positive predictions like rm_id 2761
    # (predicted ~110k, actual 0). So we keep the v3 per-track shrink and
    # only apply the INTERMITTENT override on top.
    shrinks = per_track_shrink or DEFAULT_PER_TRACK_SHRINK
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
        # Drop intermittent rm_ids from the per-rm shrink dict — they will
        # not appear in the forecast_active set and predict 0 by default.
        for rm in list(per_rm_shrink.keys()):
            if rm in intermittent_ids:
                del per_rm_shrink[rm]
    forecast_active = set(per_rm_shrink.keys())

    lin = PerRMLinearForecaster(
        fit_year=target_year - 1,
        slope_strategy="trailing_window",
        trailing_window_days=DEFAULT_TRAILING_WINDOW_DAYS,
        cutoff=history_end,
        slope_shrink=1.0,
    ).fit(ds.daily)
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
    per_track_shrink: dict[str, float] | None = None,
    use_regime: bool = True,
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
    per_track_shrink: dict[str, float] | None = None,
    use_regime: bool = True,
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

    Reverted to the v3 configuration after v4 (regime + anchor + overrides)
    showed a severe public/private LB divergence — public 4795 vs private
    16956 — confirming overfitting to year-specific patterns that don't
    generalize. v3 is the most robust configuration we have:

      - Per-rm linear regression on a 210-day trailing-window slope
      - Per-track shrink: A=0.70, B=0.70, C=0.30
      - Track D: predict 0
      - No INTERMITTENT regime override, no anchor, no manual overrides
      - Monotonicity enforced

    The v4 modules (``src/regime.py``, ``src/anchor.py``, ``src/overrides.py``)
    remain in the codebase for future experimentation but are NOT enabled
    by default. To re-run v4 manually, call ``make_submission`` with
    ``use_regime=True, use_anchor=True, manual_overrides=DEFAULT_OVERRIDES``.
    """
    ds = load_or_build()
    run = make_submission(
        ds=ds,
        use_regime=False,
        use_anchor=False,
        manual_overrides=None,
        per_track_shrink=DEFAULT_PER_TRACK_SHRINK,
        label="v5_revert_to_v3",
    )
    print("\nCV scores (lower is better):")
    for name, s in run.cv_scores.items():
        print(f"  {name}: pinball={s['mean_pinball']:.1f}")
    avg = sum(s["mean_pinball"] for s in run.cv_scores.values()) / len(run.cv_scores)
    print(f"  average: pinball={avg:.1f}")
    print("\nTrack distribution:")
    print(run.tracks["track"].value_counts().sort_index())


if __name__ == "__main__":
    main()
