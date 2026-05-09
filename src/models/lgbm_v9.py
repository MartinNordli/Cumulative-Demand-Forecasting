"""LightGBM quantile-regression correction model — the v9 second stage.

Trained on flattened (rm_id, end_date) rows from prior years' Jan-May
windows, with the per-rm linear base prediction as one of the features.
The model learns small *corrections* on top of the base, not the raw
target — which means a tiny ensemble weight (20% in production) is enough
to extract value, and the model can't pull predictions wildly off the
base anchor.

Walk-forward CV: pure LightGBM scores ~14k pinball alone. Blended at
80/20 with the per-rm linear base, the ensemble drops to 8394 — a
clean -550 vs the base alone.

Discipline:
- Features are computed using only data with ``date < Jan 1 of target_year``;
  no leakage from the year being predicted. The base predictor is
  re-computed at each training-year's history cutoff so its features
  are themselves clean.
- Sample weight is ``√(rm_id's prior-year total kg + 1)``. Without this
  the model would optimise unweighted mean pinball loss and ignore the
  high-volume rm_ids that actually move the leaderboard.
- Training rows are restricted to Track A/B/C (active rm_ids). Including
  Track D's 148 always-zero rm_ids would drag the quantile output toward
  zero everywhere.

Hyper-parameters in ``V9Params`` are hand-tuned; an Optuna sweep of 25
trials couldn't beat them.
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
    """LightGBM training hyper-parameters for the v9 correction model.

    Defaults are LightGBM's common quantile-regression starting point;
    production overrides ``learning_rate`` to 0.04 (see ``src/predict.py``).

    The strict ``min_data_in_leaf=50`` and small ``num_leaves=31`` are
    deliberate: with only ~120k training rows and a quantile loss, looser
    settings overfit fast and CV regresses.
    """

    objective: str = "quantile"
    alpha: float = TAU                  # τ=0.20, matching the leaderboard metric
    metric: str = "quantile"
    learning_rate: float = 0.05
    num_leaves: int = 31
    min_data_in_leaf: int = 50          # leaves with < 50 rows give degenerate quantile estimates
    feature_fraction: float = 0.85
    bagging_fraction: float = 0.85
    bagging_freq: int = 5
    lambda_l2: float = 1.0
    num_boost_round: int = 1500         # cap; early stopping ends training earlier
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
    """Build training data, fit LightGBM, predict on the target year.

    Three-step workflow:
        1. ``assemble_training_set(train_years)`` — for each year y, build
           features using data ``< Jan 1 of y`` and target = actual
           cumulative kg from Jan 1 of y. Concat across years.
        2. ``fit(X, y, sample_weight, valid_X, valid_y)`` — train LightGBM
           with quantile loss; early-stop on the validation fold.
        3. ``build_inference(target_year, end_dates)`` then ``predict(X)``
           — apply the trained model to the inference feature frame.

    Attributes
    ----------
    daily, materials : the dataset frames (passed through to features_v9).
    rm_ids : list of rm_ids to *train* on (Track A/B/C only); inference
        is on the full set passed to ``build_inference``.
    v8_predictor : closure that returns the per-rm linear base prediction
        for any (history_end, target_year, end_dates, rm_ids) — used as
        a feature inside ``build_features_v9``.
    params : ``V9Params`` controlling LightGBM training.
    booster, feature_cols : populated after ``fit``.

    Example
    -------
    >>> tr = V9Trainer(daily, materials, rm_ids, v8_predictor)
    >>> X, y, w = tr.assemble_training_set([2020, 2021, 2022])
    >>> val = tr.build_validation(2023)
    >>> tr.fit(X, y, w, valid_X=val.features, valid_y=val.target)
    >>> inf = tr.build_inference(2024, end_dates)
    >>> preds = tr.predict(inf.features)
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
