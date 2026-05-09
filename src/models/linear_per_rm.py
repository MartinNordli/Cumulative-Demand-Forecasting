"""Per-rm linear forecaster — the v8 base prediction.

For each rm_id we fit a single slope of cumulative kg vs day-of-year over
a trailing window (last ``trailing_window_days`` before the cutoff). The
slope is the **lower-quantile of all pairwise (i, j) slopes** — a
generalisation of the Theil-Sen estimator, which is the median of the
same set. Picking ``pair_quantile = 0.30`` gives a more conservative
slope estimate that is naturally aligned with the τ=0.20 pinball loss
that scores the leaderboard: the τ-quantile estimator is itself the
optimal point forecast for the τ-quantile loss in expectation.

The forecast is then ``slope × shrink × doy`` (multiplied by the
empirical seasonal shape, applied externally in ``predict.py``). The
shrink factor is per-track (A/B = 0.80, C = 0.50 in v8/v9) — empirically
the best on walk-forward CV.

Why this approach won across model classes (LightGBM, NHITS, …):
- Per-rm modelling captures each rm_id's individual scale without forcing
  the model to learn it from data (which fails when a single rm_id has
  10⁸ kg historical and another has 10³).
- The lower-quantile slope is robust to bursty deliveries — a single
  big-batch day doesn't blow the estimate up.
- Combined with the empirical Jan-May shape (``src/seasonality.py``),
  the forecast captures both scale and intra-window timing.

Walk-forward CV: avg pinball ≈ 8944 (v8). With LightGBM correction on top
(v9 ensemble) it drops to 8394. See README results.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd


@dataclass
class PerRMLinearForecaster:
    """Trailing-window pairwise-quantile slope estimator, per rm_id.

    Attributes
    ----------
    fit_year:
        The most recent complete year in the training data; used only for
        downstream code paths that consume yearly stats. Slopes themselves
        are fit on the trailing-day window ending at ``cutoff``.
    cutoff:
        Right-edge of the trailing window (exclusive). Typically Jan 1 of
        the forecast year.
    trailing_window_days:
        Length of the trailing window. 210 in production (v8/v9).
    pair_quantile:
        τ-quantile of pairwise slopes. 0.30 in production. Lower values
        give more conservative slopes (good for pinball-0.20).
    slope_shrink:
        Multiplicative shrink applied at predict time. Per-track override
        via ``per_rm_shrink`` is the production usage; the constant here
        is just the fallback.
    use_only_active:
        Whether to drop rows where cum_kg is still 0 from the slope fit.
    min_active_days:
        Eligibility floor — rm_ids with fewer than ``min_active_days // 4``
        active days in the window get a zero slope (they predict zero).
    pair_quantile_max_pairs:
        Cap on the number of pairwise slopes computed per rm_id, for speed.
        At 200_000 we sample uniformly when the rm_id has more pairs.
    pair_quantile_seed:
        Determinism for the pair sub-sampling above.

    Usage
    -----
    >>> m = PerRMLinearForecaster(
    ...     fit_year=2024,
    ...     cutoff=pd.Timestamp("2025-01-01"),
    ...     trailing_window_days=210,
    ...     pair_quantile=0.30,
    ... )
    >>> m.fit(daily_frame)
    >>> preds = m.predict(query_frame, per_rm_shrink={...})
    """

    fit_year: int
    cutoff: pd.Timestamp | None = None
    trailing_window_days: int = 210
    pair_quantile: float = 0.30
    slope_shrink: float = 1.0
    use_only_active: bool = True
    min_active_days: int = 20
    pair_quantile_max_pairs: int = 200_000
    pair_quantile_seed: int = 0
    fits: dict | None = None  # rm_id -> (slope, fit_succeeded)

    def fit(self, daily: pd.DataFrame) -> "PerRMLinearForecaster":
        """Fit per-rm slopes on the trailing window.

        For each rm_id, computes all unique pairwise slopes
        ``(cum_kg[j] - cum_kg[i]) / (t[j] - t[i])`` over the trailing
        window, then stores the ``pair_quantile``-th of them.
        """
        if self.cutoff is None:
            raise ValueError("PerRMLinearForecaster requires `cutoff`")

        # Restrict to the trailing window before the cutoff (cutoff exclusive).
        window_start = self.cutoff - pd.Timedelta(days=self.trailing_window_days)
        df = daily[(daily["date"] >= window_start) & (daily["date"] < self.cutoff)].copy()
        df = df.sort_values(["rm_id", "date"])
        df["t"] = (df["date"] - window_start).dt.days.astype(float)
        # Per-rm cumulative within the window — the slope target.
        df["cum_kg"] = df.groupby("rm_id")["daily_kg"].cumsum()

        rng = np.random.RandomState(self.pair_quantile_seed)
        fits: dict[int, tuple[float, bool]] = {}

        for rm_id, g in df.groupby("rm_id"):
            cum = g["cum_kg"].to_numpy(dtype=float)
            t = g["t"].to_numpy(dtype=float)
            daily_arr = g["daily_kg"].to_numpy(dtype=float)

            # Eligibility: rm_id must have at least min_active_days/4 days
            # of actual deliveries in the window. Otherwise force zero slope.
            if cum[-1] <= 0 or int((daily_arr > 0).sum()) < self.min_active_days // 4:
                fits[int(rm_id)] = (0.0, False)
                continue

            # Optionally drop pre-first-delivery rows (where cum is still 0)
            # to avoid biasing the slope downward.
            mask = np.ones_like(t, dtype=bool) if not self.use_only_active else (cum > 0)
            x = t[mask]
            y = cum[mask]
            if x.size < 2:
                fits[int(rm_id)] = (0.0, False)
                continue

            # Compute all unique pairwise slopes between time/cum-kg pairs.
            n = x.size
            i, j = np.triu_indices(n, k=1)
            # Cap memory: subsample uniformly if too many pairs.
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

            # τ-quantile of pairwise slopes is the estimator. Clamp at 0
            # because cumulative-kg slopes can't be negative.
            slope = float(max(np.quantile(slopes, self.pair_quantile), 0.0))
            fits[int(rm_id)] = (slope, True)

        self.fits = fits
        return self

    def predict(
        self,
        query: pd.DataFrame,
        rm_id_track_filter: Iterable[int] | None = None,
        per_rm_shrink: dict[int, float] | None = None,
    ) -> pd.DataFrame:
        """Forecast cumulative kg from Jan 1 to each ``forecast_end_date``.

        Parameters
        ----------
        query : DataFrame with columns ``rm_id`` and ``forecast_end_date``.
            Each row gets one prediction.
        rm_id_track_filter : optional set of rm_ids to forecast non-zero.
            rm_ids outside the set get 0.0 (used to enforce Track-D-zero).
        per_rm_shrink : optional dict {rm_id -> shrink_factor}. Per-track
            shrinks (A/B = 0.80, C = 0.50 in production) live here.

        Returns
        -------
        DataFrame [rm_id, forecast_end_date, predicted_weight].
        Predictions are clipped at 0.

        Note
        ----
        The seasonal shape (cum_kg(d) = total × shape(d)) is **not** applied
        here — that's done by the caller in ``predict.py`` so it can be
        toggled or replaced without touching this class.
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

        # Clamp to 0 since cumulative kg can't be negative.
        out["predicted_weight"] = np.maximum(0.0, shrink_arr * slope_arr * doy_arr)
        return out[["rm_id", "forecast_end_date", "predicted_weight"]]
