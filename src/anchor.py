"""Production-only: prior-year Jan-May trajectory as a conservative anchor.

For the 2025 submission we have the actual 2024 Jan-May trajectory. This
module builds an anchor prediction:

    anchor[rm_id, end_date] = alpha * cum_kg_2024_jan_may[rm_id, doy(end_date)]

which is then combined with the slope-based linear prediction via
``max(slope_pred, anchor)`` for rm_ids where the anchor is "trusted"
(prior-year H1 total ≥ ``min_anchor_kg``). For rm_ids without sufficient
prior-year H1 data, the anchor is skipped and the slope prediction is used
unchanged.

CV note: the analogous pair 2022→2023 and 2023→2024 produced *worse* CV
scores when this anchor was applied, because volumes shifted dramatically
year over year. We accept that risk for the production submission because:

1. The 2024 H1 trajectory captures within-year structure (weekly cadence,
   batch timing) that linear extrapolation cannot.
2. Many top contributors had explosive 2024 growth (3865 from 0 to 5.8M H1,
   3125 from 1.78M to 3.03M, etc.) — if 2025 stays anywhere near 2024 H1,
   the anchor outperforms the trailing-slope prediction.
3. The ``alpha = 0.65`` factor keeps us below 2024 H1 by 35%, so even if
   2025 reverts to the 2023 H1 level, we still under-predict (which is
   the cheaper side of the pinball-0.2 loss).

This module never runs during CV — ``make_submission`` calls it only when
``target_year >= 2025``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class AnchorConfig:
    alpha: float = 0.65
    min_anchor_kg: float = 50_000.0  # don't anchor on tiny prior-year volumes


def build_anchor(
    daily: pd.DataFrame,
    target_year: int,
    end_dates: pd.DatetimeIndex,
    cfg: AnchorConfig | None = None,
) -> pd.DataFrame:
    """Return a frame (rm_id, forecast_end_date, anchor_pred).

    ``anchor_pred`` is ``cfg.alpha * cum_kg_in_(target_year-1)_through_doy``
    for rm_ids whose prior-year Jan-May total is ≥ ``cfg.min_anchor_kg``.
    rm_ids with insufficient prior-year data get ``anchor_pred = 0``.
    """
    cfg = cfg or AnchorConfig()
    prior_year = target_year - 1
    df = daily[
        (daily["date"] >= pd.Timestamp(f"{prior_year}-01-01"))
        & (daily["date"] <= pd.Timestamp(f"{prior_year}-05-31"))
    ].copy()
    df = df.sort_values(["rm_id", "date"])
    df["doy"] = df["date"].dt.dayofyear
    df["cum_kg"] = df.groupby("rm_id")["daily_kg"].cumsum()

    # Eligibility: total prior-year Jan-May volume must clear the floor.
    prior_total = df.groupby("rm_id")["daily_kg"].sum()
    eligible_rms = set(prior_total[prior_total >= cfg.min_anchor_kg].index.astype(int).tolist())

    # Map (rm_id, doy) -> cumulative kg at that doy in prior year.
    cum_by_rm_doy = df.set_index(["rm_id", "doy"])["cum_kg"]

    grid = (
        pd.MultiIndex.from_product(
            [sorted(eligible_rms), end_dates], names=["rm_id", "forecast_end_date"]
        )
        .to_frame(index=False)
    )
    grid["doy"] = grid["forecast_end_date"].dt.dayofyear
    grid["cum_kg_prior"] = grid.set_index(["rm_id", "doy"]).index.map(cum_by_rm_doy)
    grid["cum_kg_prior"] = grid["cum_kg_prior"].fillna(0.0)
    grid["anchor_pred"] = cfg.alpha * grid["cum_kg_prior"]
    return grid[["rm_id", "forecast_end_date", "anchor_pred"]]


def combine_with_slope(slope_preds: pd.DataFrame, anchor_preds: pd.DataFrame) -> pd.DataFrame:
    """``max(slope_pred, anchor_pred)`` per (rm_id, end_date)."""
    merged = slope_preds.merge(
        anchor_preds, on=["rm_id", "forecast_end_date"], how="left"
    )
    merged["anchor_pred"] = merged["anchor_pred"].fillna(0.0)
    merged["predicted_weight"] = np.maximum(
        merged["predicted_weight"].to_numpy(), merged["anchor_pred"].to_numpy()
    )
    return merged[["rm_id", "forecast_end_date", "predicted_weight"]]
