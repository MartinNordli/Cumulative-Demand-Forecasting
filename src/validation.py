"""Walk-forward cross-validation that mirrors the real submission window.

Two folds: pretend-2023 (train ≤ 2022-12-31, score 2023 Jan-May) and
pretend-2024 (train ≤ 2023-12-31, score 2024 Jan-May). Pretend-2024 is
the held-out fold — tune on pretend-2023, evaluate once on pretend-2024
to estimate true generalization.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import pandas as pd

from src.data import cumulative_truth
from src.metric import mean_pinball


@dataclass
class Fold:
    name: str
    train_end: pd.Timestamp  # inclusive — training data is date < train_end + 1 day
    target_year: int  # the year whose Jan-May we score against


DEFAULT_FOLDS: list[Fold] = [
    Fold(name="pretend-2023", train_end=pd.Timestamp("2022-12-31"), target_year=2023),
    Fold(name="pretend-2024", train_end=pd.Timestamp("2023-12-31"), target_year=2024),
]


def build_query_for_fold(fold: Fold, rm_ids: Iterable[int]) -> pd.DataFrame:
    """All (rm_id, end_date) pairs in Jan 2 – May 31 of the target year."""
    end_dates = pd.date_range(
        pd.Timestamp(f"{fold.target_year}-01-02"),
        pd.Timestamp(f"{fold.target_year}-05-31"),
        freq="D",
    )
    rm_ids = list(rm_ids)
    grid = pd.MultiIndex.from_product(
        [rm_ids, end_dates], names=["rm_id", "forecast_end_date"]
    ).to_frame(index=False)
    return grid


def evaluate(
    predictions: pd.DataFrame,
    fold: Fold,
    daily: pd.DataFrame,
) -> dict:
    """Score ``predictions`` against the realised cumulative kg for ``fold``.

    ``predictions`` must have ``rm_id``, ``forecast_end_date``, ``predicted_weight``.
    """
    truth = cumulative_truth(daily, fold.target_year)
    merged = predictions.merge(
        truth, on=["rm_id", "forecast_end_date"], how="left", validate="one_to_one"
    )
    merged["actual_weight"] = merged["actual_weight"].fillna(0.0)

    losses_by_rm = (
        merged.assign(
            loss=lambda d: (
                (0.2 * (d["actual_weight"] - d["predicted_weight"])).clip(lower=0)
                + (0.8 * (d["predicted_weight"] - d["actual_weight"])).clip(lower=0)
            )
        )
        .groupby("rm_id")["loss"]
        .mean()
    )
    score = mean_pinball(merged["predicted_weight"].to_numpy(), merged["actual_weight"].to_numpy())
    return {
        "fold": fold.name,
        "mean_pinball": float(score),
        "n_rows": int(len(merged)),
        "n_rm_ids": int(merged["rm_id"].nunique()),
        "worst_rm_ids": losses_by_rm.sort_values(ascending=False).head(10).to_dict(),
    }
