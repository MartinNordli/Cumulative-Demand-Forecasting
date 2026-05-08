"""Model B — LightGBM quantile regression at τ=0.2.

Trained on flattened (rm_id, end_date) rows from prior years' Jan-May
windows (so the train distribution exactly matches inference).

Per-row sample weight: ``sqrt(total_kg_for_rm_id_in_recent_year)``. The
leaderboard score is dominated by high-volume rm_ids; without weighting,
the model would optimise unweighted mean pinball loss and ignore the
materials that actually move the needle.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

import lightgbm as lgb
import numpy as np
import pandas as pd

from src.features import FeatureBuildOutput, build_features, feature_columns
from src.metric import TAU


CATEGORICAL_FEATURES = ["rm_id", "raw_material_alloy", "raw_material_format_type"]


def _coerce_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for c in CATEGORICAL_FEATURES:
        if c in df.columns:
            df[c] = df[c].astype("category")
    return df


@dataclass
class LGBMQuantileForecaster:
    tau: float = TAU
    params: dict = field(
        default_factory=lambda: {
            "objective": "quantile",
            "alpha": TAU,
            "metric": "quantile",
            "learning_rate": 0.05,
            "num_leaves": 31,
            "min_data_in_leaf": 50,
            "feature_fraction": 0.85,
            "bagging_fraction": 0.85,
            "bagging_freq": 5,
            "lambda_l2": 1.0,
            "verbose": -1,
        }
    )
    num_boost_round: int = 1500
    early_stopping_rounds: int = 100
    booster: lgb.Booster | None = None
    feature_cols: list[str] | None = None

    def fit(
        self,
        train_df: pd.DataFrame,
        target: pd.Series,
        valid_df: pd.DataFrame | None = None,
        valid_target: pd.Series | None = None,
        sample_weight: pd.Series | None = None,
    ) -> "LGBMQuantileForecaster":
        train_df = _coerce_categoricals(train_df)
        feat_cols = feature_columns(train_df)
        self.feature_cols = feat_cols

        dtrain = lgb.Dataset(
            train_df[feat_cols],
            label=target,
            weight=sample_weight,
            categorical_feature=[c for c in CATEGORICAL_FEATURES if c in feat_cols],
            free_raw_data=False,
        )
        valid_sets = [dtrain]
        valid_names = ["train"]
        if valid_df is not None and valid_target is not None:
            valid_df = _coerce_categoricals(valid_df)
            dvalid = lgb.Dataset(
                valid_df[feat_cols],
                label=valid_target,
                categorical_feature=[c for c in CATEGORICAL_FEATURES if c in feat_cols],
                free_raw_data=False,
                reference=dtrain,
            )
            valid_sets.append(dvalid)
            valid_names.append("valid")

        self.booster = lgb.train(
            params={**self.params, "alpha": self.tau, "objective": "quantile"},
            train_set=dtrain,
            num_boost_round=self.num_boost_round,
            valid_sets=valid_sets,
            valid_names=valid_names,
            callbacks=[
                lgb.early_stopping(self.early_stopping_rounds, verbose=False),
                lgb.log_evaluation(0),
            ]
            if valid_df is not None
            else [lgb.log_evaluation(0)],
        )
        return self

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.booster is None:
            raise RuntimeError("call fit() first")
        df = _coerce_categoricals(df)
        preds = self.booster.predict(df[self.feature_cols], num_iteration=self.booster.best_iteration)
        out = df[["rm_id", "forecast_end_date"]].copy()
        out["predicted_weight"] = np.clip(preds, a_min=0.0, a_max=None)
        return out


def assemble_training_set(
    daily: pd.DataFrame,
    materials: pd.DataFrame,
    train_years: Iterable[int],
    end_dates: pd.DatetimeIndex,
    rm_ids: list[int],
    profile_for_weight: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Concatenate features+target across multiple historical years.

    ``profile_for_weight`` (optional): a per-rm summary frame; if given,
    ``sqrt(total_kg)`` is used as per-row weight.
    """
    frames: list[pd.DataFrame] = []
    targets: list[pd.Series] = []
    weights: list[pd.Series] = []
    for y in train_years:
        out: FeatureBuildOutput = build_features(daily, materials, y, end_dates, rm_ids)
        if out.target is None:
            continue
        frames.append(out.features)
        targets.append(out.target)
        if profile_for_weight is not None:
            w = (
                out.features.merge(profile_for_weight[["rm_id", "total_kg"]], on="rm_id", how="left")["total_kg"]
                .fillna(0.0)
                .clip(lower=0.0)
            )
            weights.append(np.sqrt(w + 1.0))
        else:
            weights.append(pd.Series(np.ones(len(out.features)), index=out.features.index))
    X = pd.concat(frames, ignore_index=True)
    y = pd.concat(targets, ignore_index=True)
    w = pd.concat(weights, ignore_index=True)
    return X, y, w
