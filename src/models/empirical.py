"""Model A — empirical historical 20th-percentile baseline.

For each (rm_id, day_of_year d), the forecast for the cumulative kg from
Jan 1 through day d is the τ=0.2 quantile of that same statistic across
prior years. This is the natural baseline given the metric: the optimal
point forecast under pinball-0.2 with no covariates is the marginal 20th
percentile.

Output shape: a (rm_id, doy) -> predicted_weight table that can be looked
up at any (rm_id, end_date).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.metric import TAU


@dataclass
class EmpiricalQuantileForecaster:
    tau: float = TAU
    min_year: int = 2020  # inclusive lower bound on historical years to use
    table: pd.DataFrame | None = None  # MultiIndex (rm_id, doy) -> q_cum_kg

    def fit(self, daily: pd.DataFrame, history_end: pd.Timestamp) -> "EmpiricalQuantileForecaster":
        """Fit using all complete years strictly before ``history_end``.

        Only includes years where data covers Jan 1 through May 31; partial
        years would bias the per-doy quantiles low.
        """
        df = daily[daily["date"] < history_end].copy()
        df["year"] = df["date"].dt.year
        df["doy"] = df["date"].dt.dayofyear
        df = df[df["year"] >= self.min_year]

        # Per (rm_id, year), cumulative kg through each doy from Jan 1.
        df = df.sort_values(["rm_id", "year", "doy"])
        df["cum_kg"] = df.groupby(["rm_id", "year"])["daily_kg"].cumsum()

        # Drop years that don't reach doy=151 (May 31) — partial years bias low.
        max_doy_per_year = df.groupby(["rm_id", "year"])["doy"].max().reset_index()
        complete = max_doy_per_year[max_doy_per_year["doy"] >= 151][["rm_id", "year"]]
        df = df.merge(complete, on=["rm_id", "year"], how="inner")

        # Restrict to the Jan-May horizon of interest.
        df = df[df["doy"] <= 151]

        # Per (rm_id, doy), the τ-quantile across years.
        table = (
            df.groupby(["rm_id", "doy"])["cum_kg"]
            .quantile(self.tau, interpolation="linear")
            .rename("q_cum_kg")
            .reset_index()
        )
        table["q_cum_kg"] = table["q_cum_kg"].clip(lower=0.0)

        self.table = table
        return self

    def predict(self, query: pd.DataFrame) -> pd.DataFrame:
        """Predict cumulative kg for every (rm_id, forecast_end_date) in ``query``.

        ``query`` must have columns ``rm_id`` and ``forecast_end_date``. The
        prediction is the empirical τ-quantile at doy = day-of-year of the
        end_date. rm_ids absent from the fit table predict 0 (cold-start).
        """
        if self.table is None:
            raise RuntimeError("call fit() first")
        out = query[["rm_id", "forecast_end_date"]].copy()
        out["doy"] = out["forecast_end_date"].dt.dayofyear
        out = out.merge(self.table, on=["rm_id", "doy"], how="left")
        out["q_cum_kg"] = out["q_cum_kg"].fillna(0.0)
        return out.rename(columns={"q_cum_kg": "predicted_weight"})[
            ["rm_id", "forecast_end_date", "predicted_weight"]
        ]
