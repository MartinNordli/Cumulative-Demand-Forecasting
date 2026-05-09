"""rm_id Track A/B/C/D classifier driving per-track behaviour.

The pipeline applies different shrink factors based on Track, and Track D
rm_ids are always zero — both decisions are tuned by walk-forward CV.

Track A — high-volume, very predictable: total kg in the top quartile of
   active rm_ids, active ≥ 9 months in the prior year, and R² of the
   linear fit ``cum_kg ~ doy`` ≥ 0.85. About 24 rm_ids in 2024.
Track B — mid-volume, broadly active: ≥ 6 active months. About 6 rm_ids.
Track C — sparse-but-active: had at least one delivery in H2 of the prior
   year but doesn't meet Track A or B. About 25 rm_ids; predicted with a
   smaller shrink (0.50 vs 0.80) since the slope estimate is noisier.
Track D — inactive: no H2 delivery in the prior year. About 148 rm_ids.
   Forced to zero — predicting any positive value risks the 4× over-
   prediction penalty for rm_ids that are unlikely to be active again.

The threshold values are tuned by walk-forward CV; see ``GatingThresholds``.

Inputs come from ``src.data.build_profile``: ``total_kg``,
``n_active_months``, ``had_h2_delivery``, ``linear_r2``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class GatingThresholds:
    """Thresholds defining Tracks A/B/C boundaries.

    Tuned by walk-forward CV. Track A is the strict subset (top-volume +
    sustained activity + linearity); Track B widens the activity bar;
    Track C is the catch-all for any rm_id with H2 activity in the prior
    year; rm_ids without H2 activity become Track D and predict zero.
    """

    high_volume_quantile: float = 0.75   # Track A volume floor — top quartile of active rm_ids
    high_volume_min_months: int = 9      # Track A active-months floor
    high_volume_min_r2: float = 0.85     # Track A linearity floor (R² of cum_kg vs doy)
    mid_volume_min_months: int = 6       # Track B active-months floor


def assign_tracks(
    profile: pd.DataFrame,
    thresholds: GatingThresholds | None = None,
    all_rm_ids: list[int] | None = None,
) -> pd.DataFrame:
    """Classify each rm_id into Track A/B/C/D.

    Parameters
    ----------
    profile : DataFrame from ``src.data.build_profile`` (one row per rm_id
        active in the year of interest).
    thresholds : optional ``GatingThresholds`` override.
    all_rm_ids : optional full list of rm_ids that the caller needs in the
        output. rm_ids missing from ``profile`` (i.e. inactive in the year)
        are appended as Track D so the downstream code path always sees
        every rm_id.

    Returns
    -------
    DataFrame [rm_id, track] with one row per rm_id, sorted by rm_id.
    """
    th = thresholds or GatingThresholds()
    p = profile.copy()
    if not p.empty:
        vol_cutoff = float(np.quantile(p["total_kg"].to_numpy(), th.high_volume_quantile))
    else:
        vol_cutoff = np.inf

    def classify(row) -> str:
        if not row.get("had_h2_delivery", False):
            return "D"
        # Track A — predictable + high-volume + linear cumulative curve.
        if (
            row["total_kg"] >= vol_cutoff
            and row["n_active_months"] >= th.high_volume_min_months
            and pd.notna(row["linear_r2"])
            and row["linear_r2"] >= th.high_volume_min_r2
        ):
            return "A"
        # Track B — mid volume, broadly active.
        if row["n_active_months"] >= th.mid_volume_min_months:
            return "B"
        return "C"

    p["track"] = p.apply(classify, axis=1)
    out = p[["rm_id", "track"]].copy()

    if all_rm_ids is not None:
        missing = sorted(set(all_rm_ids) - set(out["rm_id"]))
        if missing:
            out = pd.concat(
                [out, pd.DataFrame({"rm_id": missing, "track": "D"})],
                ignore_index=True,
            )
    return out.sort_values("rm_id").reset_index(drop=True)


def summarise_tracks(tracks: pd.DataFrame) -> pd.Series:
    return tracks["track"].value_counts().sort_index()
