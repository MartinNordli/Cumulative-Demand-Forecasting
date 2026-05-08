"""Data loading and daily aggregation per rm_id.

Reads the raw CSVs in ``data/`` and produces:
- ``daily``: a long frame with (rm_id, date, daily_kg) for every day from
  ``HISTORY_START`` to the last receival in 2024, zero-filled per rm_id.
- ``profile_2024``: one row per rm_id summarising 2024 behaviour, used by
  ``src.gating`` to decide which model track each rm_id is forecast on.

Cached in parquet form under ``data/processed/`` for fast reload.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

# Project paths — resolved relative to repo root.
REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data"
KERNEL_DIR = DATA_DIR / "kernel"
EXTENDED_DIR = DATA_DIR / "extended"
PROCESSED_DIR = DATA_DIR / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

# 5 stable years; older data is sparse and from a different operational regime.
HISTORY_START = pd.Timestamp("2019-01-01")
# Forecast window per the project description.
FORECAST_YEAR = 2025
FORECAST_START = pd.Timestamp(f"{FORECAST_YEAR}-01-01")
FORECAST_END = pd.Timestamp(f"{FORECAST_YEAR}-05-31")


@dataclass
class Datasets:
    daily: pd.DataFrame  # columns: rm_id, date, daily_kg
    profile_2024: pd.DataFrame  # one row per rm_id, summary stats over 2024
    materials: pd.DataFrame  # rm_id, raw_material_alloy, raw_material_format_type
    prediction_mapping: pd.DataFrame  # ID, rm_id, forecast_start_date, forecast_end_date


def _parse_arrival_date(series: pd.Series) -> pd.Series:
    # Tolerate the "+02:00" trailing tz; we just want the local calendar date.
    parsed = pd.to_datetime(series, utc=True, errors="coerce")
    return parsed.dt.tz_convert("Europe/Oslo").dt.normalize().dt.tz_localize(None)


def load_receivals(path: Path | None = None) -> pd.DataFrame:
    path = path or KERNEL_DIR / "receivals.csv"
    df = pd.read_csv(path)
    df["date"] = _parse_arrival_date(df["date_arrival"])
    df = df.dropna(subset=["date", "rm_id", "net_weight"])
    df["rm_id"] = df["rm_id"].astype(int)
    df["net_weight"] = df["net_weight"].astype(float)
    return df


def build_daily(
    receivals: pd.DataFrame,
    end_date: pd.Timestamp | None = None,
    rm_ids: list[int] | None = None,
) -> pd.DataFrame:
    """Aggregate to (rm_id, date) -> daily_kg, zero-filled over a continuous range.

    ``end_date`` controls the right edge of the index (inclusive). If ``None``,
    uses the last date in ``receivals``.

    ``rm_ids`` is the full list of rm_ids that should appear in the output —
    typically every rm_id required by the submission. rm_ids absent from
    receivals (or with deliveries only before HISTORY_START) get all-zero rows,
    which lets downstream gating assign them to Track D ("predict zero").
    """
    df = receivals[receivals["date"] >= HISTORY_START].copy()
    end_date = end_date or df["date"].max() if not df.empty else pd.Timestamp("2024-12-19")

    daily = (
        df.groupby(["rm_id", "date"], as_index=False)["net_weight"]
        .sum()
        .rename(columns={"net_weight": "daily_kg"})
    )

    # Reindex to a complete (rm_id, date) grid so cumulative sums and rolling
    # windows behave correctly even on days with no deliveries.
    full_dates = pd.date_range(HISTORY_START, end_date, freq="D")
    if rm_ids is None:
        rm_ids = sorted(daily["rm_id"].unique().tolist())
    else:
        rm_ids = sorted(set(int(x) for x in rm_ids) | set(daily["rm_id"].unique().tolist()))
    grid = pd.MultiIndex.from_product([rm_ids, full_dates], names=["rm_id", "date"])
    daily = (
        daily.set_index(["rm_id", "date"])
        .reindex(grid, fill_value=0.0)
        .reset_index()
    )
    return daily


def build_profile(daily: pd.DataFrame, year: int) -> pd.DataFrame:
    """Per-rm summary stats for the given year — used by the gating module."""
    yr = daily[daily["date"].dt.year == year].copy()
    yr["month"] = yr["date"].dt.month
    yr["doy"] = yr["date"].dt.dayofyear

    monthly = (
        yr.groupby(["rm_id", "month"])["daily_kg"].sum().reset_index()
    )
    monthly_active = monthly[monthly["daily_kg"] > 0]

    summary = (
        yr.groupby("rm_id")
        .agg(
            total_kg=("daily_kg", "sum"),
            n_active_days=("daily_kg", lambda s: int((s > 0).sum())),
            max_daily=("daily_kg", "max"),
        )
        .reset_index()
    )

    months_active = (
        monthly_active.groupby("rm_id")["month"].nunique().rename("n_active_months")
    )
    summary = summary.merge(months_active, on="rm_id", how="left")
    summary["n_active_months"] = summary["n_active_months"].fillna(0).astype(int)

    cv = (
        monthly.groupby("rm_id")["daily_kg"]
        .agg(lambda s: float(s.std() / s.mean()) if s.mean() > 0 else np.nan)
        .rename("monthly_cv")
    )
    summary = summary.merge(cv, on="rm_id", how="left")

    last_arrival = (
        yr[yr["daily_kg"] > 0].groupby("rm_id")["date"].max().rename("last_arrival")
    )
    summary = summary.merge(last_arrival, on="rm_id", how="left")
    summary["had_h2_delivery"] = summary["last_arrival"].dt.month.fillna(0).ge(7)

    # R^2 of OLS fit of cumulative kg vs day_of_year over the year — measures
    # how linearly the cumulative curve grew (Group 72's "predictability").
    cumdf = yr.copy()
    cumdf["cum_kg"] = cumdf.groupby("rm_id")["daily_kg"].cumsum()
    r2_rows = []
    for rm_id, g in cumdf.groupby("rm_id"):
        if g["cum_kg"].max() <= 0:
            r2_rows.append((rm_id, np.nan))
            continue
        x = g["doy"].to_numpy(dtype=float)
        y = g["cum_kg"].to_numpy(dtype=float)
        x_mean, y_mean = x.mean(), y.mean()
        ss_xy = ((x - x_mean) * (y - y_mean)).sum()
        ss_xx = ((x - x_mean) ** 2).sum()
        if ss_xx == 0:
            r2_rows.append((rm_id, np.nan))
            continue
        slope = ss_xy / ss_xx
        intercept = y_mean - slope * x_mean
        y_hat = slope * x + intercept
        ss_res = ((y - y_hat) ** 2).sum()
        ss_tot = ((y - y_mean) ** 2).sum()
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
        r2_rows.append((rm_id, float(r2)))
    r2_df = pd.DataFrame(r2_rows, columns=["rm_id", "linear_r2"])
    summary = summary.merge(r2_df, on="rm_id", how="left")
    return summary


def load_materials(path: Path | None = None) -> pd.DataFrame:
    """Keep only the columns we use as features (alloy + format)."""
    path = path or EXTENDED_DIR / "materials.csv"
    df = pd.read_csv(path)
    df = df.dropna(subset=["rm_id"])
    df["rm_id"] = df["rm_id"].astype(int)
    keep = ["rm_id", "raw_material_alloy", "raw_material_format_type"]
    df = df[keep].drop_duplicates(subset=["rm_id"], keep="first")
    df["raw_material_alloy"] = df["raw_material_alloy"].fillna("UNKNOWN").astype(str)
    df["raw_material_format_type"] = (
        df["raw_material_format_type"].fillna(-1).astype("Int64").astype(str)
    )
    return df


def load_prediction_mapping(path: Path | None = None) -> pd.DataFrame:
    path = path or DATA_DIR / "prediction_mapping.csv"
    df = pd.read_csv(path, parse_dates=["forecast_start_date", "forecast_end_date"])
    df["rm_id"] = df["rm_id"].astype(int)
    return df


def cumulative_truth(daily: pd.DataFrame, year: int) -> pd.DataFrame:
    """For walk-forward CV: the realised cumulative kg from Jan 1 of ``year``
    to every end_date in Jan 2 – May 31 of that year, per rm_id."""
    start = pd.Timestamp(f"{year}-01-01")
    end = pd.Timestamp(f"{year}-05-31")
    window = daily[(daily["date"] >= start) & (daily["date"] <= end)].copy()
    window["cum_kg"] = window.groupby("rm_id")["daily_kg"].cumsum()
    end_dates = pd.date_range(start + pd.Timedelta(days=1), end, freq="D")
    truth = window[window["date"].isin(end_dates)][["rm_id", "date", "cum_kg"]]
    return truth.rename(columns={"date": "forecast_end_date", "cum_kg": "actual_weight"})


def load_or_build(force: bool = False) -> Datasets:
    """One-stop entry point — loads from cache if present, else rebuilds."""
    daily_path = PROCESSED_DIR / "daily.parquet"
    profile_path = PROCESSED_DIR / "profile_2024.parquet"
    materials_path = PROCESSED_DIR / "materials.parquet"

    prediction_mapping = load_prediction_mapping()

    if not force and all(p.exists() for p in [daily_path, profile_path, materials_path]):
        daily = pd.read_parquet(daily_path)
        profile_2024 = pd.read_parquet(profile_path)
        materials = pd.read_parquet(materials_path)
    else:
        receivals = load_receivals()
        target_rm_ids = sorted(prediction_mapping["rm_id"].unique().tolist())
        daily = build_daily(receivals, rm_ids=target_rm_ids)
        profile_2024 = build_profile(daily, year=2024)
        materials = load_materials()
        daily.to_parquet(daily_path, index=False)
        profile_2024.to_parquet(profile_path, index=False)
        materials.to_parquet(materials_path, index=False)
    return Datasets(
        daily=daily,
        profile_2024=profile_2024,
        materials=materials,
        prediction_mapping=prediction_mapping,
    )


if __name__ == "__main__":
    ds = load_or_build(force=True)
    print(f"daily: {ds.daily.shape}, rm_ids={ds.daily['rm_id'].nunique()}")
    print(f"profile_2024: {ds.profile_2024.shape}")
    print(ds.profile_2024.describe())
    print(f"materials: {ds.materials.shape}")
    print(f"prediction_mapping: {ds.prediction_mapping.shape}")
    print(ds.prediction_mapping.head())
