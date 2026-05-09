"""Generate the figures used in README.md.

Produces 7 PNGs under ``docs/figures/``:
    1. score_progression.png   — CV vs LB across model versions.
    2. cv_fold_comparison.png  — pretend-2023 vs pretend-2024 per version.
    3. track_distribution.png  — rm_id counts per Track at the 2025 cutoff.
    4. seasonal_shape.png      — empirical Jan-May shape vs uniform d/151.
    5. top_loss_contributors.png — per-rm pinball on pretend-2024 (top 15).
    6. v9_predictions_top10.png — 2025 forecasts overlaid with prior years.
    7. feature_importance.png  — LightGBM gain (top 15 features).

Run with: python scripts/make_figures.py
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import matplotlib

matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Import from the project package (assumes script run from repo root).
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data import build_profile, cumulative_truth, load_or_build  # noqa: E402
from src.gating import assign_tracks  # noqa: E402
from src.metric import pinball_loss  # noqa: E402
from src.predict import (  # noqa: E402
    PER_TRACK_SHRINK,
    predict_base,
    predict_v9,
)
from src.seasonality import SeasonalityConfig, compute_shapes  # noqa: E402
from src.validation import DEFAULT_FOLDS, build_query_for_fold  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = REPO_ROOT / "docs" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)


# ----------------------------------------------------------------------
# Hard-coded score history — captured during the model evolution; used in
# figures 1 and 2. Re-generating these would require re-running every
# version's pipeline, which we no longer keep in this clean repo.
# ----------------------------------------------------------------------
SCORE_HISTORY = pd.DataFrame(
    {
        "version": ["v3", "v7", "v8", "v9"],
        "cv_avg": [9693, 9183, 8944, 8394],
        "private_lb": [9332, 8632, 7736, 7568],
        "public_lb": [7569, 6585, 5406, 5024],
        "cv_p2023": [9271, 9040, 9041, 7916],
        "cv_p2024": [10114, 9326, 8847, 8872],
    }
)


def _save(fig: plt.Figure, name: str) -> None:
    out = FIG_DIR / name
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out.relative_to(REPO_ROOT)}")


def fig_score_progression() -> None:
    """1: side-by-side CV avg vs public/private LB across versions."""
    df = SCORE_HISTORY
    x = np.arange(len(df))
    width = 0.27
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    ax.bar(x - width, df["cv_avg"], width, label="CV avg", color="#3a86ff")
    ax.bar(x, df["public_lb"], width, label="Public LB", color="#8338ec")
    ax.bar(x + width, df["private_lb"], width, label="Private LB", color="#ff006e")
    ax.set_xticks(x)
    ax.set_xticklabels(df["version"])
    ax.set_ylabel("Pinball loss (lower is better)")
    ax.set_title("Score progression across model versions")
    ax.legend(loc="upper right")
    ax.grid(axis="y", alpha=0.3)
    for i, row in df.iterrows():
        ax.text(i - width, row["cv_avg"] + 80, f"{int(row['cv_avg'])}", ha="center", fontsize=8)
        ax.text(i, row["public_lb"] + 80, f"{int(row['public_lb'])}", ha="center", fontsize=8)
        ax.text(i + width, row["private_lb"] + 80, f"{int(row['private_lb'])}", ha="center", fontsize=8)
    _save(fig, "score_progression.png")


def fig_cv_fold_comparison() -> None:
    """2: pretend-2023 vs pretend-2024 CV per version."""
    df = SCORE_HISTORY
    x = np.arange(len(df))
    width = 0.35
    fig, ax = plt.subplots(figsize=(7.5, 4.0))
    ax.bar(x - width / 2, df["cv_p2023"], width, label="pretend-2023", color="#06aed5")
    ax.bar(x + width / 2, df["cv_p2024"], width, label="pretend-2024", color="#f15bb5")
    ax.set_xticks(x)
    ax.set_xticklabels(df["version"])
    ax.set_ylabel("Pinball loss")
    ax.set_title("Walk-forward CV — both folds must improve to ship a change")
    ax.legend(loc="lower right")
    ax.grid(axis="y", alpha=0.3)
    _save(fig, "cv_fold_comparison.png")


def fig_track_distribution(ds) -> None:
    """3: count of rm_ids per Track at the 2025 prediction cutoff."""
    rm_ids = sorted(ds.daily["rm_id"].unique().tolist())
    profile = build_profile(ds.daily[ds.daily["date"] < pd.Timestamp("2025-01-01")], year=2024)
    tracks = assign_tracks(profile, all_rm_ids=rm_ids)
    counts = tracks["track"].value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(6.0, 3.5))
    colors = ["#3a86ff", "#06aed5", "#fb5607", "#999999"]
    ax.bar(counts.index, counts.values, color=colors[: len(counts)])
    for i, (lbl, v) in enumerate(counts.items()):
        ax.text(i, v + 1, str(int(v)), ha="center", fontsize=10)
    ax.set_xlabel("Track")
    ax.set_ylabel("Count of rm_ids")
    ax.set_title("Track distribution at the 2025 forecast cutoff")
    ax.grid(axis="y", alpha=0.3)
    _save(fig, "track_distribution.png")


def fig_seasonal_shape(ds) -> None:
    """4: empirical Jan-May shape vs uniform d/151."""
    daily_pre = ds.daily[ds.daily["date"] < pd.Timestamp("2025-01-01")]
    _, global_shape, _ = compute_shapes(daily_pre, fit_year=2024, cfg=SeasonalityConfig())
    doys = np.arange(1, 152)
    uniform = doys / 151.0
    fig, ax = plt.subplots(figsize=(7.5, 4.0))
    ax.plot(doys, uniform, label="Uniform (cum_kg / total = d / 151)", color="#999999", linestyle="--")
    ax.plot(doys, global_shape, label="Empirical (5y avg)", color="#ff006e", linewidth=2)
    ax.set_xlabel("Day of year (Jan 1 = 1, May 31 = 151)")
    ax.set_ylabel("Cumulative share of Jan-May total")
    ax.set_title("Why a seasonal shape matters: deliveries aren't uniform")
    # Annotate where the gap is largest.
    diff = global_shape - uniform
    max_diff_idx = int(np.argmax(np.abs(diff)))
    ax.annotate(
        f"Δ = {diff[max_diff_idx]:+.3f} at doy={max_diff_idx + 1}",
        xy=(max_diff_idx + 1, global_shape[max_diff_idx]),
        xytext=(max_diff_idx + 1 - 30, global_shape[max_diff_idx] + 0.10),
        arrowprops=dict(arrowstyle="->", color="#333"),
        fontsize=9,
    )
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)
    _save(fig, "seasonal_shape.png")


def fig_top_loss_contributors(ds) -> None:
    """5: per-rm pinball on pretend-2024 (top 15)."""
    fold = DEFAULT_FOLDS[1]  # pretend-2024
    rm_ids = sorted(ds.daily["rm_id"].unique().tolist())
    end_dates = pd.date_range(
        f"{fold.target_year}-01-02", f"{fold.target_year}-05-31", freq="D"
    )
    detail, _ = predict_v9(
        ds=ds,
        history_end=fold.train_end + pd.Timedelta(days=1),
        target_year=fold.target_year,
        end_dates=end_dates,
        rm_ids=rm_ids,
    )
    truth = cumulative_truth(ds.daily, fold.target_year)
    merged = detail.merge(truth, on=["rm_id", "forecast_end_date"], how="left")
    merged["actual_weight"] = merged["actual_weight"].fillna(0.0)
    merged["loss"] = pinball_loss(
        merged["predicted_weight"].to_numpy(),
        merged["actual_weight"].to_numpy(),
    )
    per_rm = merged.groupby("rm_id")["loss"].mean().sort_values(ascending=False)
    top = per_rm.head(15)

    fig, ax = plt.subplots(figsize=(8.5, 4.5))
    ax.barh([str(rm) for rm in top.index][::-1], top.values[::-1], color="#fb5607")
    ax.set_xlabel("Mean pinball loss per (rm_id × end_date) on pretend-2024")
    ax.set_title("Top 15 rm_ids by loss contribution — these dominate the score")
    total_loss = per_rm.sum()
    top_share = top.sum() / total_loss * 100 if total_loss else 0.0
    ax.text(
        0.98,
        0.04,
        f"Top 15 = {top_share:.1f}% of total loss",
        transform=ax.transAxes,
        fontsize=9,
        ha="right",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="#999"),
    )
    ax.grid(axis="x", alpha=0.3)
    _save(fig, "top_loss_contributors.png")


def fig_v9_predictions_top10(ds) -> None:
    """6: 2025 v9 predictions overlaid with prior-year actuals (top 10 by predicted volume)."""
    end_dates = pd.date_range("2025-01-02", "2025-05-31", freq="D")
    rm_ids = sorted(ds.daily["rm_id"].unique().tolist())
    detail, _ = predict_v9(
        ds=ds,
        history_end=pd.Timestamp("2025-01-01"),
        target_year=2025,
        end_dates=end_dates,
        rm_ids=rm_ids,
    )
    may31 = detail[detail["forecast_end_date"] == pd.Timestamp("2025-05-31")]
    top10 = may31.sort_values("predicted_weight", ascending=False).head(10)["rm_id"].tolist()

    fig, axes = plt.subplots(2, 5, figsize=(18, 7), sharex=True)
    for ax, rm in zip(axes.ravel(), top10):
        # Plot last 5 years' Jan-May cumulative.
        for year, color in zip(range(2020, 2025), plt.cm.viridis(np.linspace(0.2, 0.95, 5))):
            df = ds.daily[
                (ds.daily["rm_id"] == rm)
                & (ds.daily["date"] >= pd.Timestamp(f"{year}-01-01"))
                & (ds.daily["date"] <= pd.Timestamp(f"{year}-05-31"))
            ].sort_values("date")
            if not len(df):
                continue
            doys = df["date"].dt.dayofyear.to_numpy()
            cum = df["daily_kg"].cumsum().to_numpy()
            ax.plot(doys, cum / 1e6, label=str(year), color=color, alpha=0.75, linewidth=1.2)
        # 2025 v9 prediction.
        df_pred = detail[detail["rm_id"] == rm].sort_values("forecast_end_date")
        ax.plot(
            df_pred["forecast_end_date"].dt.dayofyear.to_numpy(),
            df_pred["predicted_weight"].to_numpy() / 1e6,
            color="#ff006e",
            linewidth=2.0,
            label="2025 (v9)",
        )
        ax.set_title(f"rm_id {rm}", fontsize=10)
        ax.grid(alpha=0.3)
        ax.set_xlabel("Day of year")
        ax.set_ylabel("Cumulative kg (M)")
    axes[0, 0].legend(loc="upper left", fontsize=8)
    fig.suptitle("v9 2025 forecast (red) vs prior-year actuals — top 10 rm_ids by predicted May 31")
    fig.tight_layout()
    _save(fig, "v9_predictions_top10.png")


def fig_feature_importance(ds) -> None:
    """7: LightGBM gain by feature, top 15."""
    # Train v9 LightGBM on the pretend-2024 fold so we have a fitted model.
    from src.features_v9 import build_features_v9
    from src.models.lgbm_v9 import V9Params, V9Trainer

    rm_ids_all = sorted(ds.daily["rm_id"].unique().tolist())
    profile = build_profile(ds.daily[ds.daily["date"] < pd.Timestamp("2024-01-01")], year=2023)
    tracks = assign_tracks(profile, all_rm_ids=rm_ids_all)
    rm_set = sorted(tracks[tracks["track"].isin(["A", "B", "C"])]["rm_id"].tolist())

    def v8_pred(history_end, target_year, end_dates, rm_ids):
        blended, _, _ = predict_base(ds, history_end, target_year, end_dates, rm_ids)
        return blended[["rm_id", "forecast_end_date", "predicted_weight"]].rename(
            columns={"predicted_weight": "v8_pred"}
        )

    params = V9Params(alpha=0.20, learning_rate=0.04, num_leaves=31, min_data_in_leaf=50, lambda_l2=1.0)
    tr = V9Trainer(daily=ds.daily, materials=ds.materials, rm_ids=rm_set, v8_predictor=v8_pred, params=params)
    train_years = [2020, 2021, 2022]
    X, y, w = tr.assemble_training_set(train_years)
    val = tr.build_validation(2023)
    tr.fit(X, y, w, valid_X=val.features, valid_y=val.target)

    importances = pd.DataFrame(
        {"feature": tr.feature_cols, "gain": tr.booster.feature_importance(importance_type="gain")}
    ).sort_values("gain", ascending=False).head(15)

    fig, ax = plt.subplots(figsize=(8.0, 5.5))
    ax.barh(importances["feature"][::-1], importances["gain"][::-1] / 1e6, color="#3a86ff")
    ax.set_xlabel("Gain (millions, summed across splits)")
    ax.set_title("LightGBM correction model — top 15 features by gain")
    ax.grid(axis="x", alpha=0.3)
    _save(fig, "feature_importance.png")


def main() -> None:
    print("Loading dataset…")
    ds = load_or_build()
    print("Generating figures →", FIG_DIR.relative_to(REPO_ROOT))
    fig_score_progression()
    fig_cv_fold_comparison()
    fig_track_distribution(ds)
    fig_seasonal_shape(ds)
    fig_top_loss_contributors(ds)
    fig_v9_predictions_top10(ds)
    fig_feature_importance(ds)
    print("All figures written.")


if __name__ == "__main__":
    main()
