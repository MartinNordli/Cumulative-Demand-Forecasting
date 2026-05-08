"""Per-doy seasonal shape multiplier.

v3 assumed that the cumulative kg through day-of-year ``d`` is exactly
``slope × d × shrink`` — uniform daily rate. v7 replaces this with

    cum_kg(d) = total_at_may31 × shape(d)

where ``shape(d)`` is the empirical normalised cumulative trajectory
averaged over the last ``n_years`` complete Jan-May windows. The May-31
prediction is unchanged; only the *intermediate* end_dates change.

Hydro's deliveries aren't uniform across Jan-May (less in January, ramp-up
through March, steady through May), and modelling that pattern improved
walk-forward CV by ~150 pinball points on both folds.

Per-rm shapes are blended with a global shape so that:
- rm_ids with rich Jan-May history (≥ ``blend_min_years``) get
  ``(1-strength) × global + strength × own``.
- rm_ids with thin or no history use the pure global shape.

This is intentionally CV-validatable — both folds run the identical code
path, no production-only behaviour.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

JAN_MAY_LAST_DOY = 151


@dataclass
class SeasonalityConfig:
    n_years: int = 5
    blend_min_years: int = 4
    blend_strength: float = 0.8


def compute_shapes(
    daily: pd.DataFrame, fit_year: int, cfg: SeasonalityConfig | None = None
) -> tuple[pd.DataFrame, np.ndarray, pd.Series]:
    """Compute per-rm and global Jan-May normalised cumulative shape.

    Returns
    -------
    per_rm_shape: DataFrame indexed by rm_id, columns = doy 1..151,
        values = mean across years of (cum_kg(y, d) / cum_kg(y, 151)).
    global_shape: ndarray length 151, the population-wide shape.
    n_active_per_rm: Series, count of years (out of ``n_years``) with
        non-zero Jan-May cumulative for the rm_id.
    """
    cfg = cfg or SeasonalityConfig()
    df = daily.copy()
    df["year"] = df["date"].dt.year
    df["doy"] = df["date"].dt.dayofyear
    df = df[
        (df["year"] >= fit_year - cfg.n_years + 1)
        & (df["year"] <= fit_year)
        & (df["doy"] <= JAN_MAY_LAST_DOY)
    ]
    df = df.sort_values(["rm_id", "year", "doy"])
    df["cum_kg"] = df.groupby(["rm_id", "year"])["daily_kg"].cumsum()
    totals = df[df["doy"] == JAN_MAY_LAST_DOY].set_index(["rm_id", "year"])["cum_kg"]
    df = df.merge(totals.rename("total"), left_on=["rm_id", "year"], right_index=True, how="left")
    df = df[df["total"] > 0]
    df["norm"] = df["cum_kg"] / df["total"]

    # Per-rm shape: average across years of normalised cum_kg at each doy.
    rm_shapes_raw = df.groupby(["rm_id", "doy"])["norm"].mean().reset_index()
    per_rm_shape = rm_shapes_raw.pivot(index="rm_id", columns="doy", values="norm")
    per_rm_shape = (
        per_rm_shape.reindex(columns=range(1, JAN_MAY_LAST_DOY + 1))
        .ffill(axis=1)
        .bfill(axis=1)
        .fillna(0.0)
    )

    n_active_per_rm = df.groupby("rm_id")["year"].nunique()

    # Global shape: average over all (rm_id, year) combinations.
    global_shape_df = (
        df.groupby("doy")["norm"]
        .mean()
        .reindex(range(1, JAN_MAY_LAST_DOY + 1))
        .ffill()
        .bfill()
        .fillna(0.0)
    )
    global_shape = global_shape_df.to_numpy()
    if global_shape[-1] > 0:
        global_shape = global_shape / global_shape[-1]
    else:
        # degenerate fallback — uniform
        global_shape = np.linspace(1 / JAN_MAY_LAST_DOY, 1.0, JAN_MAY_LAST_DOY)

    return per_rm_shape, global_shape, n_active_per_rm


def build_shape_lookup(
    eligible_rm_ids: set[int],
    per_rm_shape: pd.DataFrame,
    global_shape: np.ndarray,
    n_active_per_rm: pd.Series,
    cfg: SeasonalityConfig | None = None,
) -> dict[int, np.ndarray]:
    """Per-rm shape: blend per-rm with global by ``blend_strength`` if the
    rm_id has at least ``blend_min_years`` of valid history."""
    cfg = cfg or SeasonalityConfig()
    out: dict[int, np.ndarray] = {}
    for rm in eligible_rm_ids:
        n_active = int(n_active_per_rm.get(rm, 0)) if rm in n_active_per_rm.index else 0
        if n_active >= cfg.blend_min_years and rm in per_rm_shape.index:
            own = per_rm_shape.loc[rm].to_numpy().astype(float)
            if own[-1] > 0:
                own = own / own[-1]
            shape = (1.0 - cfg.blend_strength) * global_shape + cfg.blend_strength * own
        else:
            shape = global_shape
        out[int(rm)] = shape
    return out
