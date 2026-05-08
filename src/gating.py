"""rm_id track classification: which model(s) forecast each rm_id.

Track A — predictable, high-volume: top of total volume, active most months,
   high R^2 on the cumulative curve. Use the full ensemble (B + C + D).
Track B — predictable, mid-volume: active enough to model. Use A + B.
Track C — sparse-but-active: had at least one delivery in 2024 H2. Use A only.
Track D — inactive: no 2024 H2 delivery. Predict 0.

The profile frame comes from ``src.data.build_profile`` and includes
``total_kg``, ``n_active_months``, ``had_h2_delivery``, ``linear_r2``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class GatingThresholds:
    high_volume_quantile: float = 0.75
    high_volume_min_months: int = 9
    high_volume_min_r2: float = 0.85
    mid_volume_min_months: int = 6


def assign_tracks(
    profile: pd.DataFrame,
    thresholds: GatingThresholds | None = None,
    all_rm_ids: list[int] | None = None,
) -> pd.DataFrame:
    """Return a frame with columns rm_id, track ('A'|'B'|'C'|'D').

    ``all_rm_ids`` lets callers ensure every rm_id required by the
    submission appears in the output, even if absent from ``profile``
    (those become Track D — predict 0).
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
