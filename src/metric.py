"""Pinball (quantile) loss at τ=0.20 — the competition metric.

The asymmetric quantile loss penalises over-prediction **4× more** than
under-prediction (0.8 vs 0.2 per unit at τ=0.2):

    QuantileLoss_τ(F, A) = max( τ × (A − F),  (τ − 1) × (A − F) )
                         = max( 0.2 × (A − F),  0.8 × (F − A) )    when τ=0.2

The optimal point forecast under this loss is the **τ-quantile** of the
predictive distribution — i.e. for τ=0.2, the 20th percentile. Models
should systematically bias predictions *low* relative to the mean.

This module is the single source of truth for the metric across CV scoring,
walk-forward validation, and any ad-hoc model evaluations. The constant
``TAU`` is imported by ``src.models.lgbm_v9`` to set LightGBM's
``alpha=TAU`` (a custom quantile-loss objective at the same τ).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TAU = 0.2  # Competition metric — never change without coordinating downstream.


def pinball_loss(forecast: np.ndarray, actual: np.ndarray, tau: float = TAU) -> np.ndarray:
    """Per-row pinball loss between forecast and actual.

    Returns the same-shape array of losses. Aggregate over rows externally
    (mean for the leaderboard metric, sum for total-error analyses, etc.).
    """
    forecast = np.asarray(forecast, dtype=float)
    actual = np.asarray(actual, dtype=float)
    diff = actual - forecast
    # Under-pred (diff > 0): cost = tau * diff. Over-pred (diff < 0): cost = -(1-tau) * diff.
    # The max() form unifies both branches.
    return np.maximum(tau * diff, (tau - 1.0) * diff)


def mean_pinball(forecast: np.ndarray, actual: np.ndarray, tau: float = TAU) -> float:
    """Mean pinball loss — the leaderboard metric (averaged over all rows)."""
    return float(pinball_loss(forecast, actual, tau).mean())


def score_submission(submission: pd.DataFrame, truth: pd.DataFrame, tau: float = TAU) -> dict:
    """Compute leaderboard-style mean pinball loss.

    Both frames must contain ``ID`` and a weight column. ``truth`` uses
    ``actual_weight`` and ``submission`` uses ``predicted_weight``.
    """
    merged = submission.merge(truth, on="ID", how="inner", validate="one_to_one")
    if len(merged) != len(truth):
        missing = len(truth) - len(merged)
        raise ValueError(f"submission missing {missing} IDs from truth")

    losses = pinball_loss(
        merged["predicted_weight"].to_numpy(),
        merged["actual_weight"].to_numpy(),
        tau,
    )
    out = {
        "mean_pinball": float(losses.mean()),
        "n": int(len(merged)),
    }
    if "rm_id" in merged.columns:
        per_rm = (
            merged.assign(loss=losses)
            .groupby("rm_id")["loss"]
            .mean()
            .sort_values(ascending=False)
        )
        out["worst_rm_ids"] = per_rm.head(10).to_dict()
    return out


def _self_test() -> None:
    # F == A: zero loss
    assert pinball_loss(np.array([10.0]), np.array([10.0]))[0] == 0.0
    # Over-prediction by 1: loss = 0.8
    assert np.isclose(pinball_loss(np.array([11.0]), np.array([10.0]))[0], 0.8)
    # Under-prediction by 1: loss = 0.2
    assert np.isclose(pinball_loss(np.array([9.0]), np.array([10.0]))[0], 0.2)
    # Over-prediction by 5 vs under-prediction by 5: 4x ratio
    over = pinball_loss(np.array([15.0]), np.array([10.0]))[0]
    under = pinball_loss(np.array([5.0]), np.array([10.0]))[0]
    assert np.isclose(over / under, 4.0)
    # Vector mean
    assert np.isclose(mean_pinball(np.array([11.0, 9.0]), np.array([10.0, 10.0])), 0.5)
    print("metric self-test passed")


if __name__ == "__main__":
    _self_test()
