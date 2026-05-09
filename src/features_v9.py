"""Feature builder for the v9 LightGBM correction model.

Each row of the output frame is one (rm_id, forecast_end_date) pair. The
target is cumulative kg from Jan 1 of ``target_year`` through
``forecast_end_date``. **All features use only data with date < Jan 1 of
target_year** — zero target-year leakage.

The 25 features are grouped by what they capture:

Base prediction (1):
    ``v8_pred`` — the per-rm linear base prediction at this row. The
    LightGBM correction is given the base as a feature and learns to
    refine it, rather than predict from scratch.

Stable per-rm (8) — long-term properties; nearly invariant across years:
    ``years_active_5y``, ``total_kg_5y``, ``mean_annual_5y``,
    ``median_annual_5y``, ``std_annual_5y``, ``cv_yearly_5y``,
    ``mean_jan_may_5y``, ``median_jan_may_5y``.

Recency (5) — recent activity over windows ending at the cutoff:
    ``sum_kg_30d``, ``sum_kg_90d``, ``sum_kg_180d``, ``sum_kg_365d``,
    ``days_since_last_arrival``.

Calendar (4):
    ``doy``, ``days_into_window``, ``month``, ``day_of_week``.

Materials (2 — categorical):
    ``raw_material_alloy``, ``raw_material_format_type``.

Cross-rm pooling (3) — let the model borrow strength across similar rm_ids:
    ``alloy_group_v8_pred_mean`` (mean of v8_pred for same alloy at this
    end_date), ``format_group_v8_pred_mean`` (same for format type),
    ``alloy_group_size``.

The base predictor is **injected** as a callable so the same code path
runs in CV (with the proper history cutoff per fold) and in production.
There is no production-only behaviour anywhere in this module — the
discipline that v4 broke and v6+ enforces.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd


JAN_MAY_LAST_DOY = 151
N_YEARS_STABLE = 5

# LightGBM categorical columns.
CATEGORICAL_FEATURES = ["raw_material_alloy", "raw_material_format_type"]


def _annual_totals(daily: pd.DataFrame, years: list[int]) -> pd.DataFrame:
    """Return wide frame: index=rm_id, columns=year, values=annual_kg."""
    df = daily.copy()
    df["year"] = df["date"].dt.year
    df = df[df["year"].isin(years)]
    return df.groupby(["rm_id", "year"])["daily_kg"].sum().unstack(fill_value=0.0)


def _jan_may_totals(daily: pd.DataFrame, years: list[int]) -> pd.DataFrame:
    df = daily.copy()
    df["year"] = df["date"].dt.year
    df["doy"] = df["date"].dt.dayofyear
    df = df[df["year"].isin(years) & (df["doy"] <= JAN_MAY_LAST_DOY)]
    return df.groupby(["rm_id", "year"])["daily_kg"].sum().unstack(fill_value=0.0)


def _stable_per_rm(daily: pd.DataFrame, target_year: int) -> pd.DataFrame:
    """Per-rm summary stats over the 5-year window before ``target_year``.

    Uses only ``date < target_year`` (caller filters; we don't refilter).
    """
    years = list(range(target_year - N_YEARS_STABLE, target_year))
    annual = _annual_totals(daily, years).reindex(columns=years, fill_value=0.0)
    jm = _jan_may_totals(daily, years).reindex(columns=years, fill_value=0.0)

    out = pd.DataFrame(index=annual.index)
    out["years_active_5y"] = (annual > 0).sum(axis=1).astype(int)
    out["total_kg_5y"] = annual.sum(axis=1)
    out["mean_annual_5y"] = annual.mean(axis=1)
    out["median_annual_5y"] = annual.median(axis=1)
    out["std_annual_5y"] = annual.std(axis=1).fillna(0.0)
    out["cv_yearly_5y"] = (out["std_annual_5y"] / out["mean_annual_5y"].replace(0.0, np.nan)).fillna(0.0)
    out["mean_jan_may_5y"] = jm.mean(axis=1)
    out["median_jan_may_5y"] = jm.median(axis=1)
    return out.reset_index()


def _recency(daily: pd.DataFrame, anchor: pd.Timestamp) -> pd.DataFrame:
    """Per-rm windowed sums and days_since_last_arrival, anchored at ``anchor`` (exclusive)."""
    pre = daily[daily["date"] < anchor]
    out_index = sorted(pre["rm_id"].unique().tolist())
    out = pd.DataFrame({"rm_id": out_index}).set_index("rm_id")

    for days in (30, 90, 180, 365):
        cutoff = anchor - pd.Timedelta(days=days)
        sub = pre[(pre["date"] >= cutoff) & (pre["date"] < anchor)]
        s = sub.groupby("rm_id")["daily_kg"].sum().rename(f"sum_kg_{days}d")
        out = out.join(s, how="left")

    last = pre[pre["daily_kg"] > 0].groupby("rm_id")["date"].max()
    days_since = (anchor - last).dt.days.rename("days_since_last_arrival")
    out = out.join(days_since, how="left")
    out["days_since_last_arrival"] = out["days_since_last_arrival"].fillna(9999).astype(int)
    out = out.fillna(0.0)
    return out.reset_index()


@dataclass
class FeatureBuildOutputV9:
    features: pd.DataFrame
    target: pd.Series | None  # actual cumulative kg (NaN-able if not computable)


def build_features_v9(
    daily: pd.DataFrame,
    materials: pd.DataFrame,
    target_year: int,
    end_dates: pd.DatetimeIndex,
    rm_ids: list[int],
    v8_predictor: Callable[[pd.Timestamp, int, pd.DatetimeIndex, list[int]], pd.DataFrame],
) -> FeatureBuildOutputV9:
    """Build the v9 feature matrix for ``target_year``.

    ``v8_predictor`` signature: ``(history_end, target_year, end_dates, rm_ids) -> DataFrame``
    with columns ``rm_id, forecast_end_date, v8_pred``.
    """
    history_end = pd.Timestamp(f"{target_year}-01-01")
    daily_pre = daily[daily["date"] < history_end]

    # Base grid.
    grid = (
        pd.MultiIndex.from_product([rm_ids, end_dates], names=["rm_id", "forecast_end_date"])
        .to_frame(index=False)
    )
    grid["doy"] = grid["forecast_end_date"].dt.dayofyear.astype(int)
    grid["days_into_window"] = (grid["forecast_end_date"] - history_end).dt.days.astype(int)
    grid["month"] = grid["forecast_end_date"].dt.month.astype(int)
    grid["day_of_week"] = grid["forecast_end_date"].dt.dayofweek.astype(int)
    grid["target_year"] = int(target_year)

    # Base feature: v8 prediction.
    v8 = v8_predictor(history_end, target_year, end_dates, rm_ids)
    if "predicted_weight" in v8.columns and "v8_pred" not in v8.columns:
        v8 = v8.rename(columns={"predicted_weight": "v8_pred"})
    grid = grid.merge(v8[["rm_id", "forecast_end_date", "v8_pred"]], on=["rm_id", "forecast_end_date"], how="left")
    grid["v8_pred"] = grid["v8_pred"].fillna(0.0).clip(lower=0.0)

    # Stable per-rm.
    stable = _stable_per_rm(daily_pre, target_year)
    grid = grid.merge(stable, on="rm_id", how="left")

    # Recency.
    recency = _recency(daily_pre, history_end)
    grid = grid.merge(recency, on="rm_id", how="left")

    # Materials (categorical).
    if materials is not None and not materials.empty:
        grid = grid.merge(
            materials[["rm_id", "raw_material_alloy", "raw_material_format_type"]],
            on="rm_id",
            how="left",
        )

    # Cross-rm pooling: mean v8_pred across rm_ids in the same alloy/format
    # group at this end_date, plus group sizes. This gives LightGBM a way to
    # borrow strength across similar rm_ids.
    grid["alloy_group_v8_pred_mean"] = grid.groupby(
        ["raw_material_alloy", "forecast_end_date"]
    )["v8_pred"].transform("mean")
    grid["format_group_v8_pred_mean"] = grid.groupby(
        ["raw_material_format_type", "forecast_end_date"]
    )["v8_pred"].transform("mean")
    grid["alloy_group_size"] = grid.groupby("raw_material_alloy")["rm_id"].transform("nunique")

    # Fill remaining NaNs.
    numeric_cols = grid.select_dtypes(include=[np.number]).columns
    grid[numeric_cols] = grid[numeric_cols].fillna(0.0)
    for c in CATEGORICAL_FEATURES:
        if c in grid.columns:
            grid[c] = grid[c].fillna("UNKNOWN").astype(str)

    # Target if computable: cumulative kg from Jan 1 of target_year.
    target = _compute_target(daily, target_year, end_dates, rm_ids)
    target = (
        grid[["rm_id", "forecast_end_date"]]
        .merge(target, on=["rm_id", "forecast_end_date"], how="left")["actual_cum_kg"]
        if target is not None
        else None
    )

    return FeatureBuildOutputV9(features=grid, target=target)


def _compute_target(
    daily: pd.DataFrame, target_year: int, end_dates: pd.DatetimeIndex, rm_ids: list[int]
) -> pd.DataFrame | None:
    """Return (rm_id, forecast_end_date, actual_cum_kg) if target_year actuals are present."""
    start = pd.Timestamp(f"{target_year}-01-01")
    end = pd.Timestamp(f"{target_year}-05-31")
    avail = daily[(daily["date"] >= start) & (daily["date"] <= end)]
    if avail["date"].nunique() == 0:
        return None
    # If we have at least most of the window, build cumulative.
    df = avail.copy().sort_values(["rm_id", "date"])
    df["cum_kg"] = df.groupby("rm_id")["daily_kg"].cumsum()
    df = df.rename(columns={"date": "forecast_end_date", "cum_kg": "actual_cum_kg"})
    df = df[["rm_id", "forecast_end_date", "actual_cum_kg"]]
    df = df[df["forecast_end_date"].isin(end_dates)]
    return df


def feature_columns(features: pd.DataFrame) -> list[str]:
    """Columns to feed into LightGBM (excludes ID-like and target)."""
    drop = {"rm_id", "forecast_end_date", "target_year"}
    return [c for c in features.columns if c not in drop]
