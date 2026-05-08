"""Model D — NHiTS quantile neural model on high-volume rm_ids.

Trains a single shared NHiTS network on the daily delivery series of the
top-volume rm_ids (Track A) using ``neuralforecast`` with multi-quantile
loss ``MQLoss([0.1, 0.2, 0.5])``. The 0.2 head is used as the prediction.

Multi-quantile training stabilises the τ=0.2 head better than fitting a
single quantile alone.

Daily-delivery output is converted to cumulative via ``cumsum`` aligned to
the (rm_id, end_date) grid that the rest of the pipeline uses.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

# Lazy imports — neuralforecast is heavy and only loaded when this model is used.


def _lazy_imports():
    from neuralforecast import NeuralForecast
    from neuralforecast.losses.pytorch import MQLoss
    from neuralforecast.models import NHITS

    return NeuralForecast, MQLoss, NHITS


@dataclass
class NHITSQuantileForecaster:
    rm_ids_to_train: list[int]
    horizon: int = 151
    input_size: int = 365
    max_steps: int = 1500  # total training steps; ~1-2 min on CPU/MPS
    hidden_size: int = 128
    n_blocks: tuple = (1, 1, 1)
    n_pool_kernel_size: tuple = (2, 2, 1)
    n_freq_downsample: tuple = (24, 12, 1)
    quantiles: tuple = (0.1, 0.2, 0.5)
    target_quantile: float = 0.2
    batch_size: int = 32
    val_check_steps: int = 200
    learning_rate: float = 1e-3
    random_seed: int = 0
    accelerator: str = "cpu"  # MPS has been observed slower than CPU on small models
    devices: int | str = 1

    nf: object = None  # NeuralForecast instance
    fit_history_end: pd.Timestamp | None = None

    def _to_long_format(self, daily: pd.DataFrame) -> pd.DataFrame:
        """neuralforecast wants columns: unique_id, ds, y."""
        df = daily[daily["rm_id"].isin(self.rm_ids_to_train)].copy()
        df["unique_id"] = df["rm_id"].astype(str)
        df = df.rename(columns={"date": "ds", "daily_kg": "y"})
        return df[["unique_id", "ds", "y"]].sort_values(["unique_id", "ds"]).reset_index(drop=True)

    def fit(self, daily: pd.DataFrame, history_end: pd.Timestamp) -> "NHITSQuantileForecaster":
        NeuralForecast, MQLoss, NHITS = _lazy_imports()

        df = self._to_long_format(daily[daily["date"] < history_end])

        model = NHITS(
            h=self.horizon,
            input_size=self.input_size,
            loss=MQLoss(quantiles=list(self.quantiles)),
            n_blocks=list(self.n_blocks),
            n_pool_kernel_size=list(self.n_pool_kernel_size),
            n_freq_downsample=list(self.n_freq_downsample),
            mlp_units=[[self.hidden_size, self.hidden_size]] * 3,
            max_steps=self.max_steps,
            val_check_steps=self.val_check_steps,
            learning_rate=self.learning_rate,
            batch_size=self.batch_size,
            random_seed=self.random_seed,
            accelerator=self.accelerator,
            devices=self.devices,
            scaler_type="standard",
            enable_progress_bar=False,
        )

        self.nf = NeuralForecast(models=[model], freq="D")
        self.nf.fit(df=df)
        self.fit_history_end = history_end
        return self

    def predict_cumulative(
        self, target_year: int, end_dates: pd.DatetimeIndex
    ) -> pd.DataFrame:
        """Forecast 151 days ahead, cumsum, and align to (rm_id, end_date).

        Assumes the cutoff used at fit time is Jan 1 of ``target_year`` — i.e.
        we trained on ``daily[date < target_year-01-01]`` so the next 151 days
        start on Jan 1 of target_year.
        """
        if self.nf is None:
            raise RuntimeError("call fit() first")

        forecast = self.nf.predict()
        # neuralforecast names quantile columns ``NHITS-q-{int}`` or similar; the
        # MQLoss multi-quantile head produces a column per quantile. We pick
        # the column corresponding to target_quantile.
        col_candidates = [c for c in forecast.columns if c.startswith("NHITS")]
        # MQLoss outputs columns like 'NHITS-median', 'NHITS-lo-90', 'NHITS-hi-90'
        # but for arbitrary quantiles names depend on version. Find the
        # closest by examining numeric suffixes if present.
        target_col = None
        for c in col_candidates:
            parts = c.replace("NHITS-", "")
            if parts.endswith(f"q-{int(self.target_quantile * 100)}") or parts == f"q{int(self.target_quantile * 100)}":
                target_col = c
                break
        if target_col is None:
            # Fallback: pick the column whose mean is the lowest above zero —
            # the 0.2 quantile head is the lowest of [0.1, 0.2, 0.5] excl 0.1.
            sorted_cols = sorted(col_candidates, key=lambda c: forecast[c].mean())
            if len(sorted_cols) >= 2:
                target_col = sorted_cols[1]  # 0.2 ≈ second lowest of [0.1, 0.2, 0.5]
            else:
                target_col = sorted_cols[0]

        forecast = forecast.reset_index(drop=False) if "unique_id" not in forecast.columns else forecast
        forecast["rm_id"] = forecast["unique_id"].astype(int)
        forecast = forecast.rename(columns={"ds": "date", target_col: "daily_kg_q20"})
        forecast["daily_kg_q20"] = forecast["daily_kg_q20"].clip(lower=0.0)

        # Align: only days in [Jan 1, May 31] of target_year.
        start = pd.Timestamp(f"{target_year}-01-01")
        end = pd.Timestamp(f"{target_year}-05-31")
        forecast = forecast[(forecast["date"] >= start) & (forecast["date"] <= end)]
        forecast = forecast.sort_values(["rm_id", "date"])
        forecast["cum_kg"] = forecast.groupby("rm_id")["daily_kg_q20"].cumsum()

        out = forecast.rename(columns={"date": "forecast_end_date", "cum_kg": "predicted_weight"})
        out = out[out["forecast_end_date"].isin(end_dates)]
        return out[["rm_id", "forecast_end_date", "predicted_weight"]]
