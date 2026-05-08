"""v9 — LightGBM quantile regression refining v8 predictions.

Trained on flattened (rm_id, end_date) rows from prior years' Jan-May
windows, with the v8 prediction as a feature. The model learns *corrections*
to v8 using stable cross-rm features and alloy/format pooling.

Key discipline: every feature is computed using only data with date < Jan 1
of target_year — no leakage from the year being predicted. The v8 predictor
is recomputed at each year's history cutoff so its features are also clean.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import lightgbm as lgb
import numpy as np
import pandas as pd

from src.features_v9 import (
    CATEGORICAL_FEATURES,
    FeatureBuildOutputV9,
    build_features_v9,
    feature_columns,
)
from src.metric import TAU


def _coerce_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for c in CATEGORICAL_FEATURES:
        if c in df.columns:
            df[c] = df[c].astype("category")
    return df


@dataclass
class V9Params:
    objective: str = "quantile"
    alpha: float = TAU
    metric: str = "quantile"
    learning_rate: float = 0.05
    num_leaves: int = 31
    min_data_in_leaf: int = 50
    feature_fraction: float = 0.85
    bagging_fraction: float = 0.85
    bagging_freq: int = 5
    lambda_l2: float = 1.0
    num_boost_round: int = 1500
    early_stopping_rounds: int = 100
    verbose: int = -1
    seed: int = 0


def _to_lgb_params(p: V9Params) -> dict:
    return {
        "objective": p.objective,
        "alpha": p.alpha,
        "metric": p.metric,
        "learning_rate": p.learning_rate,
        "num_leaves": p.num_leaves,
        "min_data_in_leaf": p.min_data_in_leaf,
        "feature_fraction": p.feature_fraction,
        "bagging_fraction": p.bagging_fraction,
        "bagging_freq": p.bagging_freq,
        "lambda_l2": p.lambda_l2,
        "verbose": p.verbose,
        "seed": p.seed,
    }


@dataclass
class V9Trainer:
    """Build training data, fit LightGBM, predict on target year.

    Workflow:
        1. ``assemble_training_set(...)`` — build features+target for each
           train year and concat.
        2. ``fit(...)`` — train LightGBM with quantile loss.
        3. ``predict(...)`` — apply the trained model to a feature frame
           built for ``target_year``.
    """

    daily: pd.DataFrame
    materials: pd.DataFrame
    rm_ids: list[int]
    v8_predictor: Callable
    params: V9Params = field(default_factory=V9Params)
    booster: lgb.Booster | None = None
    feature_cols: list[str] | None = None

    def assemble_training_set(
        self,
        train_years: list[int],
    ) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
        frames, targets, weights = [], [], []
        for y in train_years:
            end_dates = pd.date_range(f"{y}-01-02", f"{y}-05-31", freq="D")
            out: FeatureBuildOutputV9 = build_features_v9(
                self.daily, self.materials, y, end_dates, self.rm_ids, self.v8_predictor
            )
            frames.append(out.features)
            targets.append(out.target.fillna(0.0))
            # Per-row weight = sqrt(rm_id total kg in year y-1 + 1).
            ann = (
                self.daily[self.daily["date"].dt.year == y - 1]
                .groupby("rm_id")["daily_kg"]
                .sum()
                .to_dict()
            )
            w = (
                out.features["rm_id"]
                .map(lambda r: np.sqrt(max(ann.get(int(r), 0.0), 0.0) + 1.0))
                .astype(float)
            )
            weights.append(w)
        X = pd.concat(frames, ignore_index=True)
        y = pd.concat(targets, ignore_index=True)
        w = pd.concat(weights, ignore_index=True)
        return X, y, w

    def build_validation(self, valid_year: int) -> FeatureBuildOutputV9:
        end_dates = pd.date_range(f"{valid_year}-01-02", f"{valid_year}-05-31", freq="D")
        return build_features_v9(
            self.daily, self.materials, valid_year, end_dates, self.rm_ids, self.v8_predictor
        )

    def build_inference(
        self, target_year: int, end_dates: pd.DatetimeIndex
    ) -> FeatureBuildOutputV9:
        return build_features_v9(
            self.daily, self.materials, target_year, end_dates, self.rm_ids, self.v8_predictor
        )

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        sample_weight: pd.Series,
        valid_X: pd.DataFrame | None = None,
        valid_y: pd.Series | None = None,
    ) -> "V9Trainer":
        X = _coerce_categoricals(X)
        feat_cols = feature_columns(X)
        self.feature_cols = feat_cols
        cat_features = [c for c in CATEGORICAL_FEATURES if c in feat_cols]

        dtrain = lgb.Dataset(
            X[feat_cols],
            label=y.astype(float),
            weight=sample_weight.astype(float),
            categorical_feature=cat_features,
            free_raw_data=False,
        )
        valid_sets = [dtrain]
        valid_names = ["train"]
        callbacks = [lgb.log_evaluation(0)]
        if valid_X is not None and valid_y is not None:
            valid_X = _coerce_categoricals(valid_X)
            dvalid = lgb.Dataset(
                valid_X[feat_cols],
                label=valid_y.fillna(0.0).astype(float),
                categorical_feature=cat_features,
                free_raw_data=False,
                reference=dtrain,
            )
            valid_sets.append(dvalid)
            valid_names.append("valid")
            callbacks = [
                lgb.early_stopping(self.params.early_stopping_rounds, verbose=False),
                lgb.log_evaluation(0),
            ]

        self.booster = lgb.train(
            params=_to_lgb_params(self.params),
            train_set=dtrain,
            num_boost_round=self.params.num_boost_round,
            valid_sets=valid_sets,
            valid_names=valid_names,
            callbacks=callbacks,
        )
        return self

    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        if self.booster is None:
            raise RuntimeError("call fit() first")
        X = _coerce_categoricals(X)
        preds = self.booster.predict(X[self.feature_cols], num_iteration=self.booster.best_iteration)
        out = X[["rm_id", "forecast_end_date"]].copy()
        out["predicted_weight"] = np.clip(preds, a_min=0.0, a_max=None)
        return out
