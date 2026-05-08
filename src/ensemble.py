"""Ensemble layer — blends per-track model outputs and applies a global
conservative shrink, monotonicity, and sanity caps.

Inputs are dicts keyed by model name, each mapping to a frame with columns
``rm_id``, ``forecast_end_date``, ``predicted_weight``. The track frame
(``rm_id`` -> ``track`` in {'A','B','C','D'}) determines per-track weights.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

DEFAULT_TRACK_WEIGHTS: dict[str, dict[str, float]] = {
    # track -> {model_name: weight}
    "A": {"linear": 0.7, "lgbm": 0.2, "nhits": 0.0, "empirical": 0.1},
    "B": {"linear": 0.5, "lgbm": 0.2, "empirical": 0.3},
    "C": {"empirical": 1.0},
    "D": {},  # predicts zero
}


@dataclass
class EnsembleConfig:
    track_weights: dict[str, dict[str, float]] = field(
        default_factory=lambda: {k: dict(v) for k, v in DEFAULT_TRACK_WEIGHTS.items()}
    )
    conservative_shrink: float = 1.0
    enforce_monotone: bool = True
    floor_zero: bool = True
    cap_multiplier: float | None = None  # cap at cap*max(prior Jan-May cum)


def blend(
    model_preds: dict[str, pd.DataFrame],
    tracks: pd.DataFrame,
    config: EnsembleConfig | None = None,
    historical_cap: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Combine model outputs into a single (rm_id, end_date, predicted_weight) frame.

    ``historical_cap``, if provided, is a frame with columns ``rm_id`` and
    ``cap`` giving the maximum prior-year Jan-May cumulative kg per rm_id.
    """
    cfg = config or EnsembleConfig()

    # All models share the same (rm_id, end_date) grid — use one as the spine.
    spine = next(iter(model_preds.values()))[["rm_id", "forecast_end_date"]].copy()
    spine = spine.sort_values(["rm_id", "forecast_end_date"]).reset_index(drop=True)

    # Materialise each model's predictions onto the spine.
    pred_matrix = {}
    for name, df in model_preds.items():
        m = spine.merge(
            df[["rm_id", "forecast_end_date", "predicted_weight"]],
            on=["rm_id", "forecast_end_date"],
            how="left",
        )["predicted_weight"].fillna(0.0).to_numpy()
        pred_matrix[name] = m

    spine = spine.merge(tracks, on="rm_id", how="left")
    spine["track"] = spine["track"].fillna("D")
    track_arr = spine["track"].to_numpy()

    final = np.zeros(len(spine))
    for track, weights in cfg.track_weights.items():
        mask = track_arr == track
        if not mask.any() or not weights:
            continue
        # Renormalise weights over the models that actually have predictions.
        active = {k: v for k, v in weights.items() if k in pred_matrix and v > 0}
        wsum = sum(active.values())
        if wsum == 0:
            continue
        for name, w in active.items():
            final[mask] += (w / wsum) * pred_matrix[name][mask]

    final *= cfg.conservative_shrink
    if cfg.floor_zero:
        final = np.maximum(final, 0.0)
    spine["predicted_weight"] = final

    # Sanity cap. rm_ids with cap == 0 (no prior Jan-May history) get np.inf
    # so the cap acts only on rm_ids where it actually has a meaningful value.
    if historical_cap is not None and cfg.cap_multiplier is not None:
        spine = spine.merge(historical_cap, on="rm_id", how="left")
        cap_col = spine["cap"].copy()
        cap_col = cap_col.where(cap_col > 0, np.inf)
        cap = (cap_col.fillna(np.inf) * cfg.cap_multiplier).to_numpy()
        spine["predicted_weight"] = np.minimum(spine["predicted_weight"].to_numpy(), cap)
        spine = spine.drop(columns=["cap"])

    # Monotonicity: cumulative is non-decreasing in end_date per rm_id.
    if cfg.enforce_monotone:
        spine = spine.sort_values(["rm_id", "forecast_end_date"]).reset_index(drop=True)
        spine["predicted_weight"] = (
            spine.groupby("rm_id")["predicted_weight"].cummax().to_numpy()
        )

    return spine[["rm_id", "forecast_end_date", "predicted_weight"]]


def historical_cap_table(daily: pd.DataFrame, max_year_inclusive: int) -> pd.DataFrame:
    """Per rm_id, the max Jan 1–May 31 cumulative kg seen in any year up to ``max_year_inclusive``."""
    df = daily[
        (daily["date"] >= pd.Timestamp("2019-01-01")) & (daily["date"].dt.year <= max_year_inclusive)
    ].copy()
    df["year"] = df["date"].dt.year
    df["doy"] = df["date"].dt.dayofyear
    df = df[df["doy"] <= 151]
    df = df.sort_values(["rm_id", "year", "date"])
    df["cum_kg"] = df.groupby(["rm_id", "year"])["daily_kg"].cumsum()
    cap = df.groupby("rm_id")["cum_kg"].max().rename("cap").reset_index()
    return cap
