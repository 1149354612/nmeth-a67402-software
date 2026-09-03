#!/usr/bin/env python
"""Generate the main-text paired ablation figure for the new 25-subject batch."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SUMMARY_DIR = ROOT / "new25_paired_ablation_plan" / "summary"
DEFAULT_OUTPUT_PNG = ROOT / "figures" / "figure3_new25_paired_ablation.png"
DEFAULT_OUTPUT_PDF = ROOT / "figures" / "figure3_new25_paired_ablation.pdf"

CONDITION_ORDER = [
    "full_mgka_h60_fc05",
    "dot_h60_fc05",
    "mgka_h60_fc00",
    "mgka_h30_fc05",
]
CONDITION_LABELS = {
    "full_mgka_h60_fc05": "Full\nMGKA H60\nFC=0.5",
    "dot_h60_fc05": "Dot\nH60\nFC=0.5",
    "mgka_h60_fc00": "MGKA\nH60\nFC=0",
    "mgka_h30_fc05": "MGKA\nH30\nFC=0.5",
}
COMPARISON_ORDER = [
    "dot_h60_fc05_vs_full_mgka_h60_fc05",
    "mgka_h60_fc00_vs_full_mgka_h60_fc05",
    "mgka_h30_fc05_vs_full_mgka_h60_fc05",
]
COMPARISON_LABELS = {
    "dot_h60_fc05_vs_full_mgka_h60_fc05": "Full - Dot",
    "mgka_h60_fc00_vs_full_mgka_h60_fc05": "Full - FC=0",
    "mgka_h30_fc05_vs_full_mgka_h60_fc05": "Full - H30",
}


def load_ablation_tables(summary_dir: Path = DEFAULT_SUMMARY_DIR) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    summary_dir = Path(summary_dir)
    per_subject = pd.read_csv(summary_dir / "new25_paired_ablation_per_subject.csv")
    condition_summary = pd.read_csv(summary_dir / "new25_paired_ablation_condition_summary.csv")
    paired = pd.read_csv(summary_dir / "new25_paired_ablation_paired_comparisons.csv")
    return per_subject, condition_summary, paired


def ci95(values: np.ndarray) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    mean = float(values.mean())
    sem = float(values.std(ddof=1) / np.sqrt(values.size))
    delta = 1.96 * sem
    return mean - delta, mean + delta


def p_label(p_value: float) -> str:
    if p_value < 0.001:
        return "p < 0.001"
    return f"p = {p_value:.3f}"


def generate_figure(
    summary_dir: Path = DEFAULT_SUMMARY_DIR,
    output_png: Path = DEFAULT_OUTPUT_PNG,
    output_pdf: Path = DEFAULT_OUTPUT_PDF,
) -> dict[str, Path]:
    import matplotlib.pyplot as plt

    per_subject, condition_summary, paired = load_ablation_tables(summary_dir)
    condition_summary = condition_summary.set_index("condition")
    paired = paired.set_index("comparison")

    rng = np.random.default_rng(20260704)
    colors = {
        "full_mgka_h60_fc05": "#1f77b4",
        "dot_h60_fc05": "#b85c5c",
        "mgka_h60_fc00": "#5a9b68",
        "mgka_h30_fc05": "#7b6bb0",
    }

    plt.rcParams.update(
        {
            "font.size": 8,
            "axes.titlesize": 9,
            "axes.labelsize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 8,
        }
    )
    fig = plt.figure(figsize=(10.6, 4.2), dpi=300, constrained_layout=True)
    grid = fig.add_gridspec(1, 2, width_ratios=[1.35, 1.0], wspace=0.32)
    ax = fig.add_subplot(grid[0, 0])
    ax_delta = fig.add_subplot(grid[0, 1])

    x = np.arange(len(CONDITION_ORDER))
    for i, condition in enumerate(CONDITION_ORDER):
        values = per_subject[condition].astype(float).to_numpy()
        jitter = rng.normal(0, 0.045, size=values.size)
        ax.scatter(
            np.full(values.size, i) + jitter,
            values,
            s=18,
            alpha=0.55,
            color=colors[condition],
            edgecolors="white",
            linewidths=0.35,
            zorder=2,
        )
        mean_value = float(condition_summary.loc[condition, "mean"])
        low, high = ci95(values)
        ax.errorbar(
            i,
            mean_value,
            yerr=[[mean_value - low], [high - mean_value]],
            fmt="o",
            markersize=7,
            color="black",
            ecolor="black",
            capsize=4,
            elinewidth=1.2,
            zorder=4,
        )
        ax.text(i, mean_value + 0.035, f"{mean_value:.3f}", ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels([CONDITION_LABELS[c] for c in CONDITION_ORDER], fontsize=8)
    ax.set_ylabel("FC reconstruction correlation (r)")
    ax.set_ylim(0.42, 1.0)
    ax.set_title("a  Subject-level FC reconstruction", loc="left", fontweight="bold", fontsize=10)
    ax.grid(axis="y", color="#d9d9d9", linewidth=0.6)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    comparison_x = np.arange(len(COMPARISON_ORDER))
    means = []
    errors = []
    for comparison in COMPARISON_ORDER:
        value = float(paired.loc[comparison, "mean_delta_full_minus_condition"])
        sd = float(paired.loc[comparison, "sd_delta_full_minus_condition"])
        n = int(paired.loc[comparison, "n"])
        sem = sd / np.sqrt(n)
        means.append(value)
        errors.append(1.96 * sem)

    bar_colors = ["#b85c5c", "#5a9b68", "#7b6bb0"]
    ax_delta.bar(comparison_x, means, yerr=errors, capsize=4, color=bar_colors, alpha=0.78, edgecolor="black", linewidth=0.6)
    ax_delta.axhline(0, color="black", linewidth=0.9)
    for i, comparison in enumerate(COMPARISON_ORDER):
        p_value = float(paired.loc[comparison, "paired_t_pvalue_bonferroni"])
        ax_delta.text(
            i,
            means[i] + (0.02 if means[i] >= 0 else -0.035),
            p_label(p_value),
            ha="center",
            va="bottom" if means[i] >= 0 else "top",
            fontsize=8,
        )
    ax_delta.set_xticks(comparison_x)
    ax_delta.set_xticklabels([COMPARISON_LABELS[c] for c in COMPARISON_ORDER], fontsize=8)
    ax_delta.set_ylabel("Paired difference in r")
    ax_delta.set_title("b  Paired full-minus-ablation differences", loc="left", fontweight="bold", fontsize=10)
    ax_delta.set_ylim(-0.09, 0.17)
    ax_delta.grid(axis="y", color="#d9d9d9", linewidth=0.6)
    ax_delta.spines["top"].set_visible(False)
    ax_delta.spines["right"].set_visible(False)

    output_png = Path(output_png)
    output_pdf = Path(output_pdf)
    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_png, dpi=300)
    fig.savefig(output_pdf)
    plt.close(fig)
    return {"png": output_png, "pdf": output_pdf}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate new-25 paired ablation figure.")
    parser.add_argument("--summary-dir", type=Path, default=DEFAULT_SUMMARY_DIR)
    parser.add_argument("--output-png", type=Path, default=DEFAULT_OUTPUT_PNG)
    parser.add_argument("--output-pdf", type=Path, default=DEFAULT_OUTPUT_PDF)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    outputs = generate_figure(args.summary_dir, args.output_png, args.output_pdf)
    print({"png": str(outputs["png"]), "pdf": str(outputs["pdf"])})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
