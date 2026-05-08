"""Manual rm_id-level overrides — Phase 5 of Plan v4.

Each override is a callable ``(detail_subset) -> Series of new predicted_weight``.
Per the plan, **at most three overrides** are allowed, each justified below
with concrete data references.

Justifications written for the production 2025 submission only — NOT
validated on CV folds (CV intentionally excludes overrides).
"""

from __future__ import annotations

import pandas as pd

from src.data import load_or_build


def _prior_h1_cum_at_doys(rm_id: int, end_dates: pd.DatetimeIndex, alpha: float) -> pd.Series:
    """Return alpha * (2024 cum_kg at each end_date's doy) for ``rm_id``.

    Used by overrides that want to anchor a prediction to last year's H1
    trajectory at a custom alpha.
    """
    ds = load_or_build()
    df = ds.daily[
        (ds.daily["rm_id"] == rm_id)
        & (ds.daily["date"] >= pd.Timestamp("2024-01-01"))
        & (ds.daily["date"] <= pd.Timestamp("2024-05-31"))
    ].copy()
    df = df.sort_values("date")
    df["doy"] = df["date"].dt.dayofyear
    df["cum_kg"] = df["daily_kg"].cumsum()
    cum_by_doy = dict(zip(df["doy"].astype(int).tolist(), df["cum_kg"].astype(float).tolist()))
    out = end_dates.dayofyear.map(lambda d: alpha * cum_by_doy.get(int(d), 0.0))
    return pd.Series(out)


def override_2130(sub: pd.DataFrame) -> pd.Series:
    """rm_id 2130: predict 0.70 × 2024 H1 cumulative trajectory.

    Justification:
        Five consecutive years of decline in Jan-May totals:
            2019: 13.6M, 2020: 12.3M, 2021: 11.2M, 2022: 12.5M, 2023: 5.4M, 2024: 3.55M
        Annual trend: 31.6M → 35.3M → 26.8M → 26.7M → 10.2M → 15.0M.
        Trailing-210d slope captures the 2024 H2 surge (11.5M) and predicts
        ~6.1M at May 31, 2025 — well above the past three years' actual.
        The pinball-0.2 cost of over-predicting at this scale is ~0.8 × 3M
        per rm_id-day. Using 0.70 of the 2024 H1 trajectory (≈2.5M at May 31)
        is more conservative and matches the recent decline trajectory.
    """
    rm = 2130
    end_dates = pd.DatetimeIndex(sub["forecast_end_date"].tolist())
    new_vals = _prior_h1_cum_at_doys(rm, end_dates, alpha=0.70)
    return new_vals.values


def override_3441(sub: pd.DataFrame) -> pd.Series:
    """rm_id 3441: predict 0.

    Justification:
        Annual deliveries: only one year of activity (2023, 3.9M Jan-May)
        followed by complete silence through 2024 (0 kg in Jan-May 2024,
        and the v4 INTERMITTENT regime already classifies it as silent).
        Explicit zero override here documents the choice and protects against
        any future logic change accidentally re-introducing a positive
        prediction. Cost-of-action: 0 (already zero).
    """
    return [0.0] * len(sub)


def override_3781(sub: pd.DataFrame) -> pd.Series:
    """rm_id 3781: predict 0.80 × 2024 H1 cumulative trajectory.

    Justification:
        Robust two-year H1 history: 2023 H1 = 6.03M, 2024 H1 = 6.53M (small
        growth, ratio 1.08). 2024 full year was 10.6M (lower than 2023's
        15.7M due to softer H2), so the trailing-210d slope under-predicts
        2025 H1 — the slope-based pred is 1.68M while anchor at α=0.65 is
        4.24M and at α=0.80 is 5.22M. The two-year H1 stability supports a
        slightly less conservative anchor than the global 0.65. With τ=0.2,
        the 0.80 value still leaves a margin against any 2025 decline.
    """
    rm = 3781
    end_dates = pd.DatetimeIndex(sub["forecast_end_date"].tolist())
    new_vals = _prior_h1_cum_at_doys(rm, end_dates, alpha=0.80)
    return new_vals.values


# Default override registry — exactly three entries, per the plan's cap.
DEFAULT_OVERRIDES = {
    2130: override_2130,
    3441: override_3441,
    3781: override_3781,
}
