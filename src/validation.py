"""Walk-forward cross-validation that mirrors the real submission window.

The competition asks for a 2025 Jan-May forecast given history through
2024-12-19. To score model changes locally we run two **walk-forward**
folds:

- **pretend-2023**: train on data ≤ 2022-12-31, score against the actual
  2023 Jan-May cumulative. Same shape as the real submission, just
  shifted back two years.
- **pretend-2024**: train on data ≤ 2023-12-31, score against 2024 Jan-May.

Walk-forward (vs random K-fold) is mandatory for time-series CV — random
splits would leak future information through the rm_id × time
correlations and give an over-optimistic score.

Discipline: every change must improve **both** folds (or at least not
regress one by more than 2%). A change that wins one fold and loses the
other is a year-specific bet, not a generalisable improvement.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import pandas as pd

from src.data import cumulative_truth
from src.metric import mean_pinball


@dataclass
class Fold:
    """A single walk-forward CV fold.

    ``train_end`` is inclusive: training data is restricted to ``date < train_end + 1 day``
    (i.e. the model never sees anything from ``target_year`` onward).
    ``target_year`` is the year whose Jan-May actuals we score against.
    """

    name: str
    train_end: pd.Timestamp
    target_year: int


# Two folds covering the most recent two complete years.
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
