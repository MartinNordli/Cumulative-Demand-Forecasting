"""Model C — per-rm linear regression on the cumulative curve.

Replicates Group 72's winning approach: for each rm_id with sufficient
history, fit a slope of cumulative kg vs day-of-year and forecast the next
year by re-applying the slope from a Jan-1 origin.

The slope is shrunk by a tunable factor ``s ∈ (0, 1]`` to bias the forecast
low (matching the τ=0.2 pinball objective). ``s`` is selected on the
walk-forward validation; expect s ≈ 0.55–0.7.

Slope strategies:
- ``"recent"``: OLS on a single year (``fit_year``).
- ``"quantile"``: OLS per year, take the τ-quantile of the slopes.
- ``"trailing_window"``: OLS on a trailing N-day window. v3 default.
- ``"trailing_window_theilsen"``: Theil-Sen (median-of-pairwise-slopes) on
  the same trailing N-day window. v7 default — substantially more robust
  to bursty deliveries than OLS.
- ``"trailing_window_qpairs"``: Lower-quantile of pairwise slopes
  (``pair_quantile``) on the trailing window — generalises Theil-Sen to
  arbitrary quantiles. With ``pair_quantile=0.30`` and matching higher
  shrink (~0.80) this beats Theil-Sen on both CV folds. v8 default.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd
from scipy.stats import theilslopes


def _fit_yearly_slope(g: pd.DataFrame, use_only_active: bool, min_active_days: int) -> tuple[float, bool]:
    cum = g["cum_kg"].to_numpy(dtype=float)
    doy = g["doy"].to_numpy(dtype=float)
    daily = g["daily_kg"].to_numpy(dtype=float)
    if cum.size == 0 or cum[-1] <= 0:
        return 0.0, False
    n_active = int((daily > 0).sum())
    if n_active < min_active_days:
        return 0.0, False
    mask = np.ones_like(doy, dtype=bool) if not use_only_active else (cum > 0)
    x = doy[mask]
    y = cum[mask]
    if x.size < 2:
        return 0.0, False
    x_mean, y_mean = x.mean(), y.mean()
    ss_xy = ((x - x_mean) * (y - y_mean)).sum()
    ss_xx = ((x - x_mean) ** 2).sum()
    if ss_xx == 0:
        return 0.0, False
    return float(ss_xy / ss_xx), True


@dataclass
class PerRMLinearForecaster:
    fit_year: int
    slope_shrink: float = 0.6
    slope_strategy: str = "recent"  # see module docstring for options
    n_years: int = 4
    quantile_tau: float = 0.5
    trailing_window_days: int = 180
    cutoff: pd.Timestamp | None = None
    use_only_active: bool = True
    min_active_days: int = 20
    pair_quantile: float = 0.50          # for "trailing_window_qpairs"
    pair_quantile_max_pairs: int = 200_000
    pair_quantile_seed: int = 0
    fits: dict | None = None

    def fit(self, daily: pd.DataFrame) -> "PerRMLinearForecaster":
        valid = (
            "recent",
            "quantile",
            "trailing_window",
            "trailing_window_theilsen",
            "trailing_window_qpairs",
        )
        if self.slope_strategy not in valid:
            raise ValueError(f"unknown slope_strategy: {self.slope_strategy}")

        if self.slope_strategy in (
            "trailing_window",
            "trailing_window_theilsen",
            "trailing_window_qpairs",
        ):
            if self.cutoff is None:
                raise ValueError("trailing_window strategy requires `cutoff`")
            window_start = self.cutoff - pd.Timedelta(days=self.trailing_window_days)
            df = daily[(daily["date"] >= window_start) & (daily["date"] < self.cutoff)].copy()
            df = df.sort_values(["rm_id", "date"])
            df["t"] = (df["date"] - window_start).dt.days.astype(float)
            df["cum_kg"] = df.groupby("rm_id")["daily_kg"].cumsum()
            rng = np.random.RandomState(self.pair_quantile_seed)
            fits: dict[int, tuple[float, bool]] = {}
            for rm_id, g in df.groupby("rm_id"):
                cum = g["cum_kg"].to_numpy(dtype=float)
                t = g["t"].to_numpy(dtype=float)
                daily_arr = g["daily_kg"].to_numpy(dtype=float)
                if cum[-1] <= 0 or int((daily_arr > 0).sum()) < self.min_active_days // 4:
                    fits[int(rm_id)] = (0.0, False)
                    continue
                mask = np.ones_like(t, dtype=bool) if not self.use_only_active else (cum > 0)
                x = t[mask]
                y = cum[mask]
                if x.size < 2:
                    fits[int(rm_id)] = (0.0, False)
                    continue
                if self.slope_strategy == "trailing_window_theilsen":
                    try:
                        slope, _, _, _ = theilslopes(y, x)
                        slope = float(max(slope, 0.0))
                    except Exception:
                        fits[int(rm_id)] = (0.0, False)
                        continue
                    fits[int(rm_id)] = (slope, True)
                elif self.slope_strategy == "trailing_window_qpairs":
                    n = x.size
                    i, j = np.triu_indices(n, k=1)
                    if i.size > self.pair_quantile_max_pairs:
                        sample = rng.choice(i.size, self.pair_quantile_max_pairs, replace=False)
                        i = i[sample]
                        j = j[sample]
                    dx = x[j] - x[i]
                    dy = y[j] - y[i]
                    valid_mask = dx > 0
                    if valid_mask.sum() < 1:
                        fits[int(rm_id)] = (0.0, False)
                        continue
                    slopes = dy[valid_mask] / dx[valid_mask]
                    slope = float(max(np.quantile(slopes, self.pair_quantile), 0.0))
                    fits[int(rm_id)] = (slope, True)
                else:
                    x_mean, y_mean = x.mean(), y.mean()
                    ss_xy = ((x - x_mean) * (y - y_mean)).sum()
                    ss_xx = ((x - x_mean) ** 2).sum()
                    if ss_xx == 0:
                        fits[int(rm_id)] = (0.0, False)
                        continue
                    fits[int(rm_id)] = (float(ss_xy / ss_xx), True)
            self.fits = fits
            return self

        df = daily.copy()
        df["year"] = df["date"].dt.year
        df["doy"] = df["date"].dt.dayofyear
        df = df.sort_values(["rm_id", "year", "doy"])
        df["cum_kg"] = df.groupby(["rm_id", "year"])["daily_kg"].cumsum()

        years = (
            [self.fit_year]
            if self.slope_strategy == "recent"
            else list(range(self.fit_year - self.n_years + 1, self.fit_year + 1))
        )
        df = df[df["year"].isin(years)]

        fits: dict[int, tuple[float, bool]] = {}
        if self.slope_strategy == "recent":
            for rm_id, g in df.groupby("rm_id"):
                slope, ok = _fit_yearly_slope(g, self.use_only_active, self.min_active_days)
                fits[int(rm_id)] = (slope, ok)
        else:
            # quantile strategy: per-rm, gather slopes across years, take quantile
            per_year_slopes: dict[int, list[float]] = {}
            for (rm_id, year), g in df.groupby(["rm_id", "year"]):
                slope, ok = _fit_yearly_slope(g, self.use_only_active, self.min_active_days)
                if ok:
                    per_year_slopes.setdefault(int(rm_id), []).append(slope)
            for rm_id, slopes in per_year_slopes.items():
                if not slopes:
                    fits[rm_id] = (0.0, False)
                    continue
                q = float(np.quantile(slopes, self.quantile_tau))
                fits[rm_id] = (q, True)
        self.fits = fits
        return self

    def predict(
        self,
        query: pd.DataFrame,
        rm_id_track_filter: Iterable[int] | None = None,
        per_rm_shrink: dict[int, float] | None = None,
    ) -> pd.DataFrame:
        """Forecast cumulative kg from Jan 1 to ``forecast_end_date`` per row.

        ``per_rm_shrink``: optional override for shrink — useful for per-track
        shrink (Track A might use 0.65 while Track B uses 0.55, say).
        """
        if self.fits is None:
            raise RuntimeError("call fit() first")
        out = query[["rm_id", "forecast_end_date"]].copy()
        out["doy"] = out["forecast_end_date"].dt.dayofyear
        s_default = self.slope_shrink

        rm_arr = out["rm_id"].to_numpy()
        doy_arr = out["doy"].to_numpy(dtype=float)
        slope_arr = np.zeros_like(doy_arr)
        shrink_arr = np.full_like(doy_arr, s_default)
        in_filter = (
            np.array([rm in rm_id_track_filter for rm in rm_arr], dtype=bool)
            if rm_id_track_filter is not None
            else np.ones_like(rm_arr, dtype=bool)
        )
        for i, rm in enumerate(rm_arr):
            fit = self.fits.get(int(rm))
            if not fit or not fit[1]:
                continue
            if not in_filter[i]:
                continue
            slope_arr[i] = fit[0]
            if per_rm_shrink is not None and int(rm) in per_rm_shrink:
                shrink_arr[i] = per_rm_shrink[int(rm)]
        out["predicted_weight"] = np.maximum(0.0, shrink_arr * slope_arr * doy_arr)
        return out[["rm_id", "forecast_end_date", "predicted_weight"]]
