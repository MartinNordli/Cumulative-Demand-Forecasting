"""Feature engineering for the cumulative-kg target.

Each row corresponds to a tuple (rm_id, end_date, year_to_predict) where
``year_to_predict`` is the year whose Jan 1 → end_date cumulative kg we
are forecasting. **All features use only data before Jan 1 of
year_to_predict** to avoid leakage; the target itself is computable when
the actuals through end_date are available.

Calling pattern:
- training: pass several ``year_to_predict`` values (e.g. 2020..2023) so
  the model sees multiple years of (features, target) pairs.
- inference: pass just the forecast year (2025 for the final submission,
  2023 / 2024 for walk-forward validation).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

# Days into the Jan-May window for each end_date — bounded by 0..151.
JAN1_WINDOW_DAYS = 151


def _historical_cum_by_doy(daily: pd.DataFrame, max_year: int, min_year: int = 2019) -> pd.DataFrame:
    """Per (rm_id, year, doy), the cumulative kg from Jan 1.

    Restricted to ``min_year <= year <= max_year`` so the caller can
    exclude the forecast year cleanly.
    """
    df = daily[
        (daily["date"].dt.year >= min_year) & (daily["date"].dt.year <= max_year)
    ].copy()
    df["year"] = df["date"].dt.year
    df["doy"] = df["date"].dt.dayofyear
    df = df[df["doy"] <= JAN1_WINDOW_DAYS]
    df = df.sort_values(["rm_id", "year", "doy"])
    df["cum_kg"] = df.groupby(["rm_id", "year"])["daily_kg"].cumsum()
    return df[["rm_id", "year", "doy", "cum_kg"]]


def _recency_features(daily: pd.DataFrame, anchor: pd.Timestamp) -> pd.DataFrame:
    """Per rm_id, summary stats on the deliveries strictly before ``anchor``."""
    pre = daily[daily["date"] < anchor].copy()
    if pre.empty:
        return pd.DataFrame(columns=["rm_id"])

    def windowed(d: pd.Timestamp, days: int) -> pd.DataFrame:
        cutoff = d - pd.Timedelta(days=days)
        sub = pre[(pre["date"] >= cutoff) & (pre["date"] < d)]
        return (
            sub.groupby("rm_id")
            .agg(
                **{
                    f"sum_kg_{days}d": ("daily_kg", "sum"),
                    f"active_days_{days}d": (
                        "daily_kg",
                        lambda s: int((s > 0).sum()),
                    ),
                }
            )
            .reset_index()
        )

    f30 = windowed(anchor, 30)
    f90 = windowed(anchor, 90)
    f180 = windowed(anchor, 180)
    f365 = windowed(anchor, 365)
    out = f30.merge(f90, on="rm_id", how="outer")
    out = out.merge(f180, on="rm_id", how="outer")
    out = out.merge(f365, on="rm_id", how="outer")

    last_arrival = (
        pre[pre["daily_kg"] > 0].groupby("rm_id")["date"].max().rename("last_arrival_date")
    )
    out = out.merge(last_arrival, on="rm_id", how="outer")
    out["days_since_last_arrival"] = (anchor - out["last_arrival_date"]).dt.days
    out["days_since_last_arrival"] = out["days_since_last_arrival"].fillna(9999).astype(int)
    out = out.drop(columns=["last_arrival_date"])
    return out.fillna(0)


def _slope_last_90d(daily: pd.DataFrame, anchor: pd.Timestamp) -> pd.DataFrame:
    """OLS slope of cumulative kg vs day index over the last 90 days."""
    cutoff = anchor - pd.Timedelta(days=90)
    sub = daily[(daily["date"] >= cutoff) & (daily["date"] < anchor)].copy()
    if sub.empty:
        return pd.DataFrame(columns=["rm_id", "slope_90d"])
    sub = sub.sort_values(["rm_id", "date"])
    sub["t"] = (sub["date"] - cutoff).dt.days.astype(float)
    sub["cum_kg"] = sub.groupby("rm_id")["daily_kg"].cumsum()

    def fit(g: pd.DataFrame) -> float:
        x = g["t"].to_numpy()
        y = g["cum_kg"].to_numpy()
        if x.size < 5:
            return 0.0
        ss_xx = ((x - x.mean()) ** 2).sum()
        if ss_xx == 0:
            return 0.0
        return float(((x - x.mean()) * (y - y.mean())).sum() / ss_xx)

    rows = [(rm_id, fit(g)) for rm_id, g in sub.groupby("rm_id")]
    return pd.DataFrame(rows, columns=["rm_id", "slope_90d"])


def _historical_pattern_features(
    daily: pd.DataFrame, year_to_predict: int, end_dates: pd.DatetimeIndex
) -> pd.DataFrame:
    """For each (rm_id, end_date), historical cum_kg at the same day-of-year.

    Output columns:
      cum_y{2019..year_to_predict-1}, hist_mean_5y, hist_median_5y,
      hist_q20_5y, hist_q20_3y, hist_yoy_ratio (last vs prior).
    """
    hist = _historical_cum_by_doy(daily, max_year=year_to_predict - 1)
    if hist.empty:
        return pd.DataFrame()

    # Pivot: index=(rm_id, doy), columns=year, values=cum_kg.
    wide = (
        hist.pivot_table(index=["rm_id", "doy"], columns="year", values="cum_kg")
        .reset_index()
    )
    wide.columns.name = None

    year_cols = [c for c in wide.columns if isinstance(c, (int, np.integer))]
    year_cols_sorted = sorted(year_cols)
    last3 = year_cols_sorted[-3:]
    # Forward-fill within each (rm_id, year) over doy so missing doy use the
    # last known cumulative — required because zero-fill may have produced
    # gaps if a year had no deliveries before that doy.
    for c in year_cols:
        wide[c] = wide[c].astype(float).fillna(0.0)

    wide["hist_mean_all"] = wide[year_cols].mean(axis=1)
    wide["hist_median_all"] = wide[year_cols].median(axis=1)
    wide["hist_q20_all"] = wide[year_cols].quantile(0.2, axis=1)
    wide["hist_q20_last3"] = wide[last3].quantile(0.2, axis=1) if len(last3) >= 1 else 0.0
    wide["hist_max_all"] = wide[year_cols].max(axis=1)
    if len(year_cols_sorted) >= 2:
        last_year = year_cols_sorted[-1]
        prior_year = year_cols_sorted[-2]
        denom = wide[prior_year].replace(0.0, np.nan)
        wide["hist_yoy_ratio"] = (wide[last_year] / denom).replace([np.inf, -np.inf], np.nan).fillna(1.0).clip(0, 5)
    else:
        wide["hist_yoy_ratio"] = 1.0

    # Rename year columns for clarity.
    wide = wide.rename(columns={c: f"cum_y{c}" for c in year_cols})

    # Restrict to requested doys.
    wanted_doys = pd.Series(end_dates.dayofyear).unique()
    wide = wide[wide["doy"].isin(wanted_doys)]
    return wide


@dataclass
class FeatureBuildOutput:
    features: pd.DataFrame  # (rm_id, forecast_end_date, year_to_predict, ...features...)
    target: pd.Series | None  # cum_kg from Jan 1 of year_to_predict to end_date, if available


def build_features(
    daily: pd.DataFrame,
    materials: pd.DataFrame,
    year_to_predict: int,
    end_dates: pd.DatetimeIndex,
    rm_ids: list[int],
) -> FeatureBuildOutput:
    """Build a (rm_id × end_date) feature frame for ``year_to_predict``.

    All features use data with date < Jan 1 of year_to_predict.
    """
    anchor = pd.Timestamp(f"{year_to_predict}-01-01")

    # Base grid.
    grid = pd.MultiIndex.from_product(
        [rm_ids, end_dates], names=["rm_id", "forecast_end_date"]
    ).to_frame(index=False)
    grid["year_to_predict"] = year_to_predict
    grid["doy"] = grid["forecast_end_date"].dt.dayofyear
    grid["days_into_window"] = (grid["forecast_end_date"] - anchor).dt.days
    grid["month"] = grid["forecast_end_date"].dt.month
    grid["day_of_week"] = grid["forecast_end_date"].dt.dayofweek
    grid["week_of_year"] = grid["forecast_end_date"].dt.isocalendar().week.astype(int)
    grid["is_august"] = (grid["month"] == 8).astype(int)

    # Historical pattern features (per rm_id × doy).
    hist = _historical_pattern_features(daily, year_to_predict, end_dates)
    if not hist.empty:
        grid = grid.merge(hist, on=["rm_id", "doy"], how="left")

    # Recency features (per rm_id, single anchor=Jan 1 of forecast year).
    rec = _recency_features(daily, anchor)
    if not rec.empty:
        grid = grid.merge(rec, on="rm_id", how="left")

    slope_df = _slope_last_90d(daily, anchor)
    if not slope_df.empty:
        grid = grid.merge(slope_df, on="rm_id", how="left")

    # Static features from materials.
    if materials is not None and not materials.empty:
        grid = grid.merge(materials, on="rm_id", how="left")

    # Default fills for any rm_id with no history.
    numeric_cols = grid.select_dtypes(include=[np.number]).columns
    grid[numeric_cols] = grid[numeric_cols].fillna(0.0)
    for c in ["raw_material_alloy", "raw_material_format_type"]:
        if c in grid:
            grid[c] = grid[c].fillna("UNKNOWN").astype("category")

    # Target if computable: cumulative kg in [Jan 1, end_date].
    target: pd.Series | None
    needed_dates = pd.date_range(anchor, end_dates.max(), freq="D")
    available = (
        daily[(daily["date"] >= anchor) & (daily["date"] <= end_dates.max())]
    )
    if available["date"].nunique() == len(needed_dates):
        target_df = available.copy()
        target_df = target_df.sort_values(["rm_id", "date"])
        target_df["cum_kg"] = target_df.groupby("rm_id")["daily_kg"].cumsum()
        target_df = target_df.rename(columns={"date": "forecast_end_date"})
        merged = grid[["rm_id", "forecast_end_date"]].merge(
            target_df[["rm_id", "forecast_end_date", "cum_kg"]],
            on=["rm_id", "forecast_end_date"],
            how="left",
        )
        merged["cum_kg"] = merged["cum_kg"].fillna(0.0)
        target = merged["cum_kg"].astype(float)
    else:
        target = None

    return FeatureBuildOutput(features=grid, target=target)


def feature_columns(features: pd.DataFrame) -> list[str]:
    """Columns to feed into LightGBM (excludes IDs, end_date, target)."""
    drop = {"rm_id", "forecast_end_date", "year_to_predict"}
    return [c for c in features.columns if c not in drop]
