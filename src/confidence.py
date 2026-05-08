"""Per-rm confidence score → slope shrink.

Confidence is derived from MULTI-YEAR stable features, never from the
target year. The same code path runs in CV (target_year ∈ {2023, 2024})
and in production (target_year = 2025). This is the discipline that the
v4 regime/anchor/overrides violated: those layers were either CV-excluded
or used target-year information.

Inputs:
- ``years_active_5y``: number of distinct years (in the 5-year window
  ending at ``cutoff.year - 1``) with at least one delivery.
- ``yearly_total_cv_5y``: coefficient of variation of annual kg totals.
  Low CV ⇒ steady rm_id; high CV ⇒ erratic.
- ``total_kg_5y``: cumulative kg over the 5y window — distinguishes
  "active" from "long-since-inactive" rm_ids.

Output: ``shrink ∈ {0.0, 0.30, 0.40, 0.65, 0.85}``.

Design note: HIGH gets 0.85 (less conservative) because 5 stable years
makes the recent slope trustworthy. LOW gets 0.40 because high CV means
the recent slope might be a fluke. INACTIVE gets 0.0 to preserve the
v3 behaviour of zero-predicting Track-D-equivalent rm_ids.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class ConfidenceThresholds:
    n_years_window: int = 5
    high_min_active: int = 5         # all 5 years had deliveries
    high_max_cv: float = 0.5
    mid_min_active: int = 4
    mid_max_cv: float = 1.0
    low_min_active: int = 2
    inactive_max_total_kg: float = 0.0  # strictly zero kg over the window


SHRINK_MAP: dict[str, float] = {
    "HIGH": 0.85,
    "MID": 0.65,
    "LOW": 0.40,
    "NEW": 0.30,
    "INACTIVE": 0.0,
}


def _annual_totals(daily: pd.DataFrame, years: list[int]) -> pd.DataFrame:
    """Return wide frame: index=rm_id, columns=year, values=annual_kg."""
    df = daily.copy()
    df["year"] = df["date"].dt.year
    df = df[df["year"].isin(years)]
    return df.groupby(["rm_id", "year"])["daily_kg"].sum().unstack(fill_value=0.0)


def confidence_features(daily: pd.DataFrame, cutoff: pd.Timestamp, n_years: int = 5) -> pd.DataFrame:
    """Compute per-rm confidence features over the 5-year window before ``cutoff``."""
    pre = daily[daily["date"] < cutoff].copy()
    fit_year = cutoff.year - 1
    years = list(range(fit_year - n_years + 1, fit_year + 1))
    annual = _annual_totals(pre, years)
    annual = annual.reindex(columns=years, fill_value=0.0)

    out = pd.DataFrame(index=annual.index)
    out["years_active_5y"] = (annual > 0).sum(axis=1).astype(int)
    out["total_kg_5y"] = annual.sum(axis=1)
    # CV computed only over years with > 0 kg to avoid degenerate zeros.
    def cv_active(row: pd.Series) -> float:
        active = row[row > 0]
        if len(active) < 2:
            return float("nan")
        m = active.mean()
        if m <= 0:
            return float("nan")
        return float(active.std() / m)

    out["yearly_total_cv_5y"] = annual.apply(cv_active, axis=1)
    out = out.reset_index()
    return out


def classify_confidence(
    daily: pd.DataFrame,
    cutoff: pd.Timestamp,
    th: ConfidenceThresholds | None = None,
) -> pd.DataFrame:
    th = th or ConfidenceThresholds()
    feats = confidence_features(daily, cutoff, n_years=th.n_years_window)

    def label(row) -> str:
        if row["total_kg_5y"] <= th.inactive_max_total_kg:
            return "INACTIVE"
        if row["years_active_5y"] >= th.high_min_active and (
            pd.notna(row["yearly_total_cv_5y"]) and row["yearly_total_cv_5y"] < th.high_max_cv
        ):
            return "HIGH"
        if row["years_active_5y"] >= th.mid_min_active and (
            pd.notna(row["yearly_total_cv_5y"]) and row["yearly_total_cv_5y"] < th.mid_max_cv
        ):
            return "MID"
        if row["years_active_5y"] >= th.low_min_active:
            return "LOW"
        return "NEW"

    feats["band"] = feats.apply(label, axis=1)
    feats["shrink"] = feats["band"].map(SHRINK_MAP)
    return feats


def shrink_per_rm(
    confidence: pd.DataFrame, restrict_to: set[int] | None = None
) -> dict[int, float]:
    df = confidence
    if restrict_to is not None:
        df = df[df["rm_id"].isin(restrict_to)]
    return {int(r["rm_id"]): float(r["shrink"]) for _, r in df.iterrows() if r["shrink"] > 0}
