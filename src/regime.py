"""Per-rm YoY-trend regime classifier.

Each rm_id is assigned a regime that determines its slope shrink. Designed
to address two concrete failure modes in the v3 pipeline:

1. Persistent under-prediction on growing/stable rm_ids (3125, 3282, 3122,
   3124, 3123, 3126, ...) where the global 0.7 shrink is too aggressive.
2. False-positive predictions on rm_ids that delivered late in year T-1 but
   went silent (e.g. 2761: predicted 111k, actual 0).

Thresholds are set a priori from EDA — *not* tuned on the eval fold — to
avoid leaking the target year into the regime boundaries.

Inputs to ``classify_regime``:
- ``daily``: full long frame (rm_id, date, daily_kg). Filter to ``< cutoff``
  inside this module so the same call works for CV folds and production.
- ``cutoff``: ``pd.Timestamp`` — first day of the forecast year (e.g.
  ``2025-01-01`` for production, ``2024-01-01`` for pretend-2024 fold).

Outputs: a frame with one row per rm_id and columns ``regime`` and
``shrink``. Use ``shrink_per_rm(...)`` to convert to the dict expected by
``PerRMLinearForecaster.predict(per_rm_shrink=...)``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

# Regime → slope-shrink prior. Each value is a small *adjustment* around the
# proven v3 default of 0.7, not a wholesale departure. Aggressive shifts (e.g.
# 0.95 / 0.45) over-fit one year of YoY signal and regress badly on the other
# fold. The user-confirmed "moderate (0.85–0.95)" range for GROWING is
# respected at the upper bound (0.85).
REGIME_SHRINK: dict[str, float] = {
    "GROWING": 0.85,    # +0.15 vs default
    "STABLE": 0.75,     # +0.05 vs default
    "DECLINING": 0.55,  # -0.15 vs default
    "NEW": 0.65,        # -0.05 vs default
    "INTERMITTENT": 0.0,  # forces zero prediction
    "DEFAULT": 0.70,    # current v3 default
}


@dataclass
class RegimeThresholds:
    grow_jm_yoy_min: float = 1.15
    grow_h2_yoy_min: float = 1.0
    grow_intensity_min: float = 0.10  # 90d window, not 30d — Dec/holidays would tank a 30d intensity

    stable_jm_yoy_low: float = 0.85
    stable_jm_yoy_high: float = 1.15
    stable_intensity_min: float = 0.10

    decline_jm_yoy_max: float = 0.7
    decline_h2_yoy_max: float = 0.5

    # INTERMITTENT (false-positive guard) — must be unambiguous: long silence
    # since last delivery and very low recent volume. Healthy seasonal rm_ids
    # (December slowdown) must not trigger this.
    intermittent_days_since_last_delivery_min: int = 60
    intermittent_sum_last_90d_max: float = 50_000.0
    intermittent_active_days_last_90d_max: int = 4


def _per_year_window_total(
    daily: pd.DataFrame, year: int, doy_low: int, doy_high: int
) -> pd.Series:
    """Sum of daily_kg per rm_id for [doy_low, doy_high] in ``year``.

    Returns an indexed-by-rm_id series, zero-filled for rm_ids absent that year.
    """
    df = daily[daily["date"].dt.year == year].copy()
    df["doy"] = df["date"].dt.dayofyear
    df = df[(df["doy"] >= doy_low) & (df["doy"] <= doy_high)]
    return df.groupby("rm_id")["daily_kg"].sum()


def _windowed(daily: pd.DataFrame, anchor: pd.Timestamp, days: int, agg: str) -> pd.Series:
    """Per-rm aggregation over ``[anchor - days, anchor)``.

    ``agg`` ∈ {"sum_kg", "active_days"}.
    """
    cutoff = anchor - pd.Timedelta(days=days)
    sub = daily[(daily["date"] >= cutoff) & (daily["date"] < anchor)]
    if agg == "sum_kg":
        return sub.groupby("rm_id")["daily_kg"].sum()
    if agg == "active_days":
        return sub.assign(active=sub["daily_kg"] > 0).groupby("rm_id")["active"].sum().astype(int)
    raise ValueError(f"unknown agg: {agg}")


def trend_signals(daily: pd.DataFrame, cutoff: pd.Timestamp) -> pd.DataFrame:
    """Compute the per-rm features used by the regime classifier.

    All features use only ``daily[daily['date'] < cutoff]`` — no leak.
    """
    df = daily[daily["date"] < cutoff].copy()
    rm_ids = df["rm_id"].unique()
    out = pd.DataFrame({"rm_id": rm_ids}).set_index("rm_id")

    fit_year = cutoff.year - 1  # most recent complete year
    prior_year = fit_year - 1

    # Jan-May totals (doy 1..151) for the last two complete years.
    jm_fit = _per_year_window_total(df, fit_year, 1, 151)
    jm_prev = _per_year_window_total(df, prior_year, 1, 151)
    out["jm_fit"] = jm_fit.reindex(out.index).fillna(0.0)
    out["jm_prev"] = jm_prev.reindex(out.index).fillna(0.0)
    # YoY ratio with safe divide; if prev=0 → NaN (the NEW path picks this up).
    out["jm_yoy"] = (out["jm_fit"] / out["jm_prev"].replace(0.0, np.nan))

    # H2 totals (doy 183..366) — proxy for late-year acceleration.
    h2_fit = _per_year_window_total(df, fit_year, 183, 366)
    h2_prev = _per_year_window_total(df, prior_year, 183, 366)
    out["h2_fit"] = h2_fit.reindex(out.index).fillna(0.0)
    out["h2_prev"] = h2_prev.reindex(out.index).fillna(0.0)
    out["h2_yoy"] = (out["h2_fit"] / out["h2_prev"].replace(0.0, np.nan))

    # Recent activity — use 90d as the primary "recent" window so that
    # one slow month (e.g. Norwegian December shutdowns) doesn't wrongly
    # flag a high-volume rm_id as intermittent.
    sum_90 = _windowed(df, cutoff, 90, "sum_kg").reindex(out.index).fillna(0.0)
    sum_120 = _windowed(df, cutoff, 120, "sum_kg").reindex(out.index).fillna(0.0)
    active_90 = _windowed(df, cutoff, 90, "active_days").reindex(out.index).fillna(0).astype(int)

    out["sum_kg_last_90d"] = sum_90
    out["sum_kg_last_120d"] = sum_120
    out["active_days_last_90d"] = active_90
    out["recent_intensity"] = active_90 / 90.0  # active days fraction over the last 90d

    # Days since last delivery — the cleanest signal for "went silent".
    last_arrival = (
        df[df["daily_kg"] > 0].groupby("rm_id")["date"].max().rename("last_arrival_date")
    )
    last_arrival_aligned = last_arrival.reindex(out.index)
    out["days_since_last_delivery"] = (cutoff - last_arrival_aligned).dt.days.fillna(9999).astype(int)

    # Total kg in fit year (used for NEW detection / sanity)
    out["total_fit_year"] = df[df["date"].dt.year == fit_year].groupby("rm_id")["daily_kg"].sum().reindex(out.index).fillna(0.0)
    out["total_prev_year"] = df[df["date"].dt.year == prior_year].groupby("rm_id")["daily_kg"].sum().reindex(out.index).fillna(0.0)

    return out.reset_index()


def classify_regime(
    daily: pd.DataFrame,
    cutoff: pd.Timestamp,
    thresholds: RegimeThresholds | None = None,
) -> pd.DataFrame:
    """Assign each rm_id a regime label and per-rm shrink.

    Order matters — first match wins:
      1) INTERMITTENT (false-positive guard)
      2) NEW (no prior Jan-May, but fit-year deliveries)
      3) DECLINING
      4) GROWING
      5) STABLE
      6) DEFAULT
    """
    th = thresholds or RegimeThresholds()
    sigs = trend_signals(daily, cutoff)

    def label(row) -> str:
        # 1) INTERMITTENT — gone silent. Triple-condition guard so that
        # a healthy rm_id with a December slowdown does NOT trigger:
        #   - last delivery was over ``intermittent_days_since_last_delivery_min`` days ago
        #   - very low last-90d volume
        #   - very few last-90d active days
        if (
            row["days_since_last_delivery"] >= th.intermittent_days_since_last_delivery_min
            and row["sum_kg_last_90d"] <= th.intermittent_sum_last_90d_max
            and row["active_days_last_90d"] <= th.intermittent_active_days_last_90d_max
        ):
            return "INTERMITTENT"

        # 2) NEW — no prior Jan-May, but fit-year had deliveries.
        if row["jm_prev"] == 0 and row["total_fit_year"] > 0:
            return "NEW"

        # 3) DECLINING — sharply lower than prior year (Jan-May or H2).
        if (pd.notna(row["jm_yoy"]) and row["jm_yoy"] <= th.decline_jm_yoy_max) or (
            pd.notna(row["h2_yoy"]) and row["h2_yoy"] <= th.decline_h2_yoy_max
        ):
            return "DECLINING"

        # 4) GROWING — both Jan-May and H2 trending up, with sustained recent activity.
        if (
            pd.notna(row["jm_yoy"])
            and row["jm_yoy"] >= th.grow_jm_yoy_min
            and pd.notna(row["h2_yoy"])
            and row["h2_yoy"] >= th.grow_h2_yoy_min
            and row["recent_intensity"] >= th.grow_intensity_min
        ):
            return "GROWING"

        # 5) STABLE — close-to-1 YoY and active.
        if (
            pd.notna(row["jm_yoy"])
            and th.stable_jm_yoy_low <= row["jm_yoy"] <= th.stable_jm_yoy_high
            and row["recent_intensity"] >= th.stable_intensity_min
        ):
            return "STABLE"

        return "DEFAULT"

    sigs["regime"] = sigs.apply(label, axis=1)
    sigs["shrink"] = sigs["regime"].map(REGIME_SHRINK)
    return sigs


def shrink_per_rm(
    regimes: pd.DataFrame, restrict_to: set[int] | None = None
) -> dict[int, float]:
    """Convert a regime frame into the dict consumed by PerRMLinearForecaster."""
    if restrict_to is not None:
        df = regimes[regimes["rm_id"].isin(restrict_to)]
    else:
        df = regimes
    return dict(zip(df["rm_id"].astype(int), df["shrink"].astype(float)))


def _self_test() -> None:
    """Synthetic test: build a tiny 'daily' frame with known patterns.

    Each rm_id is constructed so its 2023→2024 YoY clearly hits one regime.
    """
    rng = pd.date_range("2022-01-01", "2024-12-31", freq="D")
    rows = []
    # (rm_id) → (kg_per_day_2022, kg_per_day_2023, kg_per_day_2024, kind)
    rm_specs = {
        1: (100, 100, 200, "growing"),       # +100% YoY in 2024
        2: (200, 200, 60, "declining"),       # -70% YoY in 2024
        3: (50, 50, 50, "intermittent"),      # zeroed out for last 90d
        4: (0, 0, 200, "new"),                # no prior, big in 2024
        5: (100, 100, 105, "stable"),         # ~5% growth
    }
    for rm, (k22, k23, k24, kind) in rm_specs.items():
        for d in rng:
            yr_kg = {2022: k22, 2023: k23, 2024: k24}[d.year]
            k = float(yr_kg)
            if kind == "intermittent" and (d >= pd.Timestamp("2024-09-01")):
                k = 0.0
            if kind == "new" and d < pd.Timestamp("2024-01-01"):
                k = 0.0
            rows.append({"rm_id": rm, "date": d, "daily_kg": k})
    daily = pd.DataFrame(rows)
    daily["date"] = pd.to_datetime(daily["date"])

    res = classify_regime(daily, pd.Timestamp("2025-01-01"))
    print(res.set_index("rm_id")[["jm_yoy", "h2_yoy", "recent_intensity", "regime", "shrink"]])
    expected = {1: "GROWING", 2: "DECLINING", 3: "INTERMITTENT", 4: "NEW", 5: "STABLE"}
    actual = dict(zip(res["rm_id"], res["regime"]))
    for rm, want in expected.items():
        got = actual.get(rm)
        assert got == want, f"rm {rm}: expected {want}, got {got}"
    print("regime self-test passed")


if __name__ == "__main__":
    _self_test()
