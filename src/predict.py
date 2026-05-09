"""End-to-end prediction pipeline for the v9 ensemble.

The v9 model is an ensemble of two complementary predictors:

1.  **Base** (per-rm linear, ``src/models/linear_per_rm.py``): for each rm_id,
    a single slope of cumulative kg vs day-of-year is estimated from the
    last 210 days of history using the τ=0.30 quantile of pairwise
    slopes. The slope is multiplied by the empirical Jan-May
    seasonal shape (``src/seasonality.py``) and a per-track shrink
    (A=B=0.80, C=0.50; Track D rm_ids predict zero).

2.  **Correction** (LightGBM quantile, ``src/models/lgbm_v9.py``): a small
    gradient-booster trained with α=0.20 quantile loss on stable
    cross-rm features (5y mean/median/std/cv, recency windows, alloy
    + format pooling). The base prediction itself is one of its features,
    so it learns a correction rather than the raw target.

The final v9 prediction is ``0.80 × base + 0.20 × correction``. The blend
weight, the LightGBM hyper-parameters, the slope quantile, the seasonal
shape, and the per-track shrinks are all tuned by walk-forward CV (two
folds: pretend-2023 and pretend-2024) under the strict discipline that
both folds must improve before any change is shipped.

Walk-forward CV pinball: avg 8394 (p2023=7916, p2024=8872).
Leaderboard: private 7568, public 5024.

Entry points:
    python -m src.predict      → produces submission CSV + prints CV
    src.predict.make_submission(ds=..., label="...") → programmatic.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.data import (
    REPO_ROOT,
    Datasets,
    build_profile,
    load_or_build,
)
from src.gating import assign_tracks
from src.models.linear_per_rm import PerRMLinearForecaster
from src.seasonality import SeasonalityConfig, build_shape_lookup, compute_shapes
from src.validation import DEFAULT_FOLDS, evaluate

SUBMISSIONS_DIR = REPO_ROOT / "submissions"
SUBMISSIONS_DIR.mkdir(parents=True, exist_ok=True)

# Per-track slope shrinks, picked on walk-forward CV.
# A and B are the high- and mid-volume rm_ids; C is sparse-but-active.
# Track D (no H2 delivery) always predicts zero.
PER_TRACK_SHRINK: dict[str, float] = {"A": 0.80, "B": 0.80, "C": 0.50}

TRAILING_WINDOW_DAYS = 210
PAIR_QUANTILE = 0.30  # The τ-quantile of pairwise slopes — naturally aligned with the τ=0.20 metric.

# v9 ensemble blend weight: 80% per-rm linear base + 20% LightGBM correction.
W_BASE = 0.80
W_LGBM = 0.20

# LightGBM hyper-parameters for the correction model. Hand-tuned;
# Optuna couldn't find better in 25 trials.
LGBM_LR = 0.04
LGBM_NUM_LEAVES = 31
LGBM_MIN_DATA_IN_LEAF = 50
LGBM_ALPHA = 0.20
LGBM_LAMBDA_L2 = 1.0


@dataclass
class PredictionRun:
    """Output bundle of a v9 prediction run.

    Attributes
    ----------
    submission : DataFrame [ID, predicted_weight] — the CSV-ready submission.
    detail : DataFrame [rm_id, forecast_end_date, predicted_weight, track]
        — same predictions but keyed by rm_id/end_date for inspection.
    tracks : DataFrame [rm_id, track] — the gating decision used.
    cv_scores : dict {fold_name -> {mean_pinball, ...}} — walk-forward CV.
    """

    submission: pd.DataFrame
    detail: pd.DataFrame
    tracks: pd.DataFrame
    cv_scores: dict


def predict_base(
    ds: Datasets,
    history_end: pd.Timestamp,
    target_year: int,
    end_dates: pd.DatetimeIndex,
    rm_ids: list[int],
) -> tuple[pd.DataFrame, pd.DataFrame, set[int]]:
    """Compute the v8-style **base** prediction (slope × seasonal shape × shrink).

    This is the foundation of v9: the LightGBM correction sits on top of
    these predictions, so this function is called both during the v9
    pipeline (for inference) and from inside ``src/features_v9.py`` (to
    expose ``v8_pred`` as a feature for the correction model).

    Returns
    -------
    blended : DataFrame [rm_id, forecast_end_date, predicted_weight].
    tracks : DataFrame [rm_id, track] from the gating module.
    forecast_active : set of rm_ids that are predicted non-zero.
    """
    daily_pre = ds.daily[ds.daily["date"] < history_end]

    # Track A/B/C/D classification — Track D predicts zero, A/B/C use the slope.
    profile = build_profile(daily_pre, year=target_year - 1)
    tracks = assign_tracks(profile, all_rm_ids=rm_ids)
    track_a = set(tracks[tracks["track"] == "A"]["rm_id"].tolist())
    track_b = set(tracks[tracks["track"] == "B"]["rm_id"].tolist())
    track_c = set(tracks[tracks["track"] == "C"]["rm_id"].tolist())
    forecast_active = track_a | track_b | track_c

    per_rm_shrink = {
        **{rm: PER_TRACK_SHRINK["A"] for rm in track_a},
        **{rm: PER_TRACK_SHRINK["B"] for rm in track_b},
        **{rm: PER_TRACK_SHRINK["C"] for rm in track_c},
    }

    # Fit per-rm slopes on the last 210 days using the q=0.30 pairwise-slope estimator.
    lin = PerRMLinearForecaster(
        fit_year=target_year - 1,
        cutoff=history_end,
        trailing_window_days=TRAILING_WINDOW_DAYS,
        pair_quantile=PAIR_QUANTILE,
    ).fit(ds.daily)

    # Compute the empirical Jan-May seasonal shape (rm-specific where data exists,
    # blended toward a global shape for sparse rm_ids).
    per_rm_shape, global_shape, n_active_per_rm = compute_shapes(
        daily_pre, fit_year=target_year - 1, cfg=SeasonalityConfig()
    )
    shape_lookup = build_shape_lookup(
        forecast_active, per_rm_shape, global_shape, n_active_per_rm, SeasonalityConfig()
    )

    # Base prediction: total_at_may31 × shape(d). At d=151 the shape is 1.0,
    # so the May-31 prediction equals (slope × 151 × shrink). At intermediate
    # doys the shape captures Hydro's actual delivery cadence (not uniform).
    query = (
        pd.MultiIndex.from_product([rm_ids, end_dates], names=["rm_id", "forecast_end_date"])
        .to_frame(index=False)
    )
    rm_arr = query["rm_id"].to_numpy()
    doy_arr = query["forecast_end_date"].dt.dayofyear.to_numpy()
    slope_arr = np.array(
        [
            (lin.fits.get(int(rm), (0.0, False))[0]
             if lin.fits.get(int(rm), (0.0, False))[1]
             else 0.0)
            for rm in rm_arr
        ]
    )
    shrink_arr = np.array([per_rm_shrink.get(int(rm), 0.0) for rm in rm_arr])
    active_mask = np.array([int(rm) in forecast_active for rm in rm_arr])
    total_pred = slope_arr * 151.0 * shrink_arr * active_mask

    shape_vals = np.array(
        [
            shape_lookup.get(int(rm), global_shape)[int(d) - 1] if 1 <= int(d) <= 151 else 1.0
            for rm, d in zip(rm_arr, doy_arr)
        ]
    )
    blended = query[["rm_id", "forecast_end_date"]].copy()
    blended["predicted_weight"] = np.maximum(0.0, total_pred * shape_vals)

    # Enforce monotonicity per rm_id (cumulative is non-decreasing in time).
    blended = blended.sort_values(["rm_id", "forecast_end_date"]).reset_index(drop=True)
    blended["predicted_weight"] = blended.groupby("rm_id")["predicted_weight"].cummax().to_numpy()

    blended = blended.merge(tracks, on="rm_id", how="left")
    return blended, tracks, forecast_active


def _v8_predictor_for_features(
    ds: Datasets,
) -> "callable":
    """Return a closure that v9's feature builder uses to inject ``v8_pred``.

    The LightGBM correction model takes the base prediction as one of its
    inputs (so it learns to refine, not predict from scratch). For each
    training year y, the feature builder needs the base prediction
    *as if* the cutoff were Jan 1 of year y — this closure carries the
    dataset and forwards the call to ``predict_base``.
    """

    def predictor(history_end, target_year, end_dates, rm_ids):
        blended, _, _ = predict_base(ds, history_end, target_year, end_dates, rm_ids)
        return blended[["rm_id", "forecast_end_date", "predicted_weight"]].rename(
            columns={"predicted_weight": "v8_pred"}
        )

    return predictor


def _apply_lgbm_correction(
    ds: Datasets,
    base_preds: pd.DataFrame,
    target_year: int,
    end_dates: pd.DatetimeIndex,
    rm_ids: list[int],
    forecast_active: set[int],
) -> pd.DataFrame:
    """Train a quantile LightGBM on prior years and blend with the base.

    Training years are the three years preceding ``target_year`` (e.g. for
    target=2025: train on 2021/2022/2023, validate on 2024 for early
    stopping). The model is restricted to Track A/B/C rm_ids — including
    the long tail of inactive Track D rm_ids would drag the LightGBM's
    quantile output toward zero. Track D stays at zero either way.

    Final prediction is ``W_BASE × base + W_LGBM × lgbm`` (80/20 in
    production). LightGBM rarely outperforms the base on its own, but at
    20% weight it consistently lowers CV by ~550 pinball points by
    catching cases where the base is over-extrapolating on declining
    rm_ids.
    """
    # Lazy import to avoid loading LightGBM unless we actually run a v9 pass.
    from src.models.lgbm_v9 import V9Params, V9Trainer

    train_years = [target_year - 4, target_year - 3, target_year - 2]
    valid_year = target_year - 1
    rm_set = sorted(forecast_active)
    if not rm_set:
        return base_preds

    params = V9Params(
        alpha=LGBM_ALPHA,
        learning_rate=LGBM_LR,
        num_leaves=LGBM_NUM_LEAVES,
        min_data_in_leaf=LGBM_MIN_DATA_IN_LEAF,
        lambda_l2=LGBM_LAMBDA_L2,
    )
    tr = V9Trainer(
        daily=ds.daily,
        materials=ds.materials,
        rm_ids=rm_set,
        v8_predictor=_v8_predictor_for_features(ds),
        params=params,
    )
    X, y, w = tr.assemble_training_set(train_years)
    val = tr.build_validation(valid_year)
    tr.fit(X, y, w, valid_X=val.features, valid_y=val.target)

    # Inference grid covers ALL rm_ids; rm_ids outside forecast_active get zeroed below.
    tr.rm_ids = rm_ids
    inf = tr.build_inference(target_year, end_dates)
    lgbm_preds = tr.predict(inf.features).rename(columns={"predicted_weight": "lgbm"})

    out = base_preds.copy()
    out = out.merge(lgbm_preds, on=["rm_id", "forecast_end_date"], how="left")
    out["lgbm"] = out["lgbm"].fillna(0.0)
    out["predicted_weight"] = W_BASE * out["predicted_weight"] + W_LGBM * out["lgbm"]
    out.loc[~out["rm_id"].isin(forecast_active), "predicted_weight"] = 0.0
    out["predicted_weight"] = out["predicted_weight"].clip(lower=0.0)

    # Re-enforce monotonicity after the blend (cummax per rm_id).
    out = out.sort_values(["rm_id", "forecast_end_date"]).reset_index(drop=True)
    out["predicted_weight"] = out.groupby("rm_id")["predicted_weight"].cummax().to_numpy()
    return out.drop(columns=["lgbm"])


def predict_v9(
    ds: Datasets,
    history_end: pd.Timestamp,
    target_year: int,
    end_dates: pd.DatetimeIndex,
    rm_ids: list[int],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Full v9 prediction at a given history cutoff.

    Returns
    -------
    detail : DataFrame [rm_id, forecast_end_date, predicted_weight, track]
    tracks : DataFrame [rm_id, track] from the gating module.
    """
    base_preds, tracks, forecast_active = predict_base(
        ds, history_end, target_year, end_dates, rm_ids
    )
    detail = _apply_lgbm_correction(
        ds=ds,
        base_preds=base_preds,
        target_year=target_year,
        end_dates=end_dates,
        rm_ids=rm_ids,
        forecast_active=forecast_active,
    )
    if "track" not in detail.columns:
        detail = detail.merge(tracks, on="rm_id", how="left")
    return detail, tracks


def cv_score(ds: Datasets, rm_ids: list[int]) -> dict:
    """Run walk-forward CV (pretend-2023 + pretend-2024) end-to-end.

    Each fold trains v9 from scratch on its own history cutoff and scores
    against that fold's actuals. Returns a dict keyed by fold name with
    per-fold pinball scores and worst-rm_id breakdown.
    """
    out: dict[str, dict] = {}
    for fold in DEFAULT_FOLDS:
        end_dates = pd.date_range(
            f"{fold.target_year}-01-02", f"{fold.target_year}-05-31", freq="D"
        )
        detail, _ = predict_v9(
            ds=ds,
            history_end=fold.train_end + pd.Timedelta(days=1),
            target_year=fold.target_year,
            end_dates=end_dates,
            rm_ids=rm_ids,
        )
        out[fold.name] = evaluate(
            detail[["rm_id", "forecast_end_date", "predicted_weight"]], fold, ds.daily
        )
    return out


def make_submission(ds: Datasets, label: str | None = None) -> PredictionRun:
    """Train v9 on full history, score CV, and emit the production submission.

    Parameters
    ----------
    ds : Datasets — typically from ``load_or_build()``.
    label : optional string used in the submission filename.

    Returns
    -------
    PredictionRun bundle. Side effect: writes ``submissions/submission_<label>_<ts>.csv``
    if ``label`` is provided.
    """
    rm_ids = sorted(ds.daily["rm_id"].unique().tolist())
    pm = ds.prediction_mapping
    end_dates = pd.DatetimeIndex(sorted(pm["forecast_end_date"].unique()))

    # Production prediction: history through 2024, target 2025.
    detail, tracks = predict_v9(
        ds=ds,
        history_end=pd.Timestamp("2025-01-01"),
        target_year=2025,
        end_dates=end_dates,
        rm_ids=rm_ids,
    )

    # Map predictions onto the (ID, predicted_weight) submission grid.
    submission = pm.merge(
        detail[["rm_id", "forecast_end_date", "predicted_weight"]],
        on=["rm_id", "forecast_end_date"],
        how="left",
    )
    submission["predicted_weight"] = submission["predicted_weight"].fillna(0.0).clip(lower=0.0)
    submission = submission[["ID", "predicted_weight"]].sort_values("ID").reset_index(drop=True)

    cv_scores = cv_score(ds=ds, rm_ids=rm_ids)
    sanity_checks(submission, detail, ds.prediction_mapping)

    if label:
        ts = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        path = SUBMISSIONS_DIR / f"submission_{label}_{ts}.csv"
        submission.to_csv(path, index=False)
        print(f"wrote {path}")

    return PredictionRun(
        submission=submission, detail=detail, tracks=tracks, cv_scores=cv_scores
    )


def sanity_checks(
    submission: pd.DataFrame, detail: pd.DataFrame, prediction_mapping: pd.DataFrame
) -> None:
    """Verify the submission is well-formed before writing.

    Checks: row count matches the prediction mapping, every ID is covered,
    no negative predicted weights, and predicted_weight is monotonically
    non-decreasing in forecast_end_date for each rm_id (since cumulative
    deliveries can only grow).
    """
    assert len(submission) == len(prediction_mapping), (
        f"row count mismatch: {len(submission)} vs {len(prediction_mapping)}"
    )
    assert set(submission["ID"]) == set(prediction_mapping["ID"]), "ID set mismatch"
    assert (submission["predicted_weight"] >= 0).all(), "negative predictions present"

    issues = (
        detail.sort_values(["rm_id", "forecast_end_date"])
        .groupby("rm_id")["predicted_weight"]
        .apply(lambda s: bool((s.diff().dropna() < -1e-6).any()))
    )
    bad = issues[issues]
    if not bad.empty:
        raise AssertionError(
            f"non-monotonic predictions for {len(bad)} rm_ids: {bad.index.tolist()[:5]}"
        )
    print(f"sanity OK: {len(submission)} rows, all non-negative, monotonic per rm_id")


def main() -> None:
    """Build the v9 production submission.

    Walk-forward CV (avg pinball, lower better):
        v3 9693, v7 9183, v8 8944, **v9 8394**.
    Leaderboard: private 7568, public 5024.
    """
    ds = load_or_build()
    run = make_submission(ds=ds, label="v9_ensemble_lgbm")
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
