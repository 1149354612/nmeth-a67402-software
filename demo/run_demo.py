#!/usr/bin/env python
"""Run a lightweight simulated-data demo for the reviewer software package."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def load_timeseries(path: Path) -> np.ndarray:
    array = np.load(path)
    if array.ndim != 2:
        raise ValueError(f"Expected a 2D array, got shape {array.shape}")
    if array.shape[1] != 360:
        raise ValueError(f"Expected 360 columns, got shape {array.shape}")
    if not np.isfinite(array).all():
        raise ValueError("Demo data contains NaN or Inf values")
    return array.astype(np.float64, copy=False)


def compute_summary(ts: np.ndarray) -> tuple[dict[str, object], pd.DataFrame, pd.DataFrame]:
    n_timepoints, n_regions = ts.shape
    region_means = ts.mean(axis=0)
    region_stds = ts.std(axis=0, ddof=0)
    safe_stds = np.where(region_stds == 0.0, 1.0, region_stds)
    standardized = (ts - region_means) / safe_stds
    corr = np.corrcoef(standardized, rowvar=False)

    off_diag = corr[np.triu_indices(n_regions, k=1)]
    preview = corr[:10, :10]

    summary = {
        "n_timepoints": int(n_timepoints),
        "n_regions": int(n_regions),
        "global_mean": float(ts.mean()),
        "global_std": float(ts.std(ddof=0)),
        "region_0_1_correlation": float(corr[0, 1]),
        "mean_absolute_offdiag_correlation": float(np.mean(np.abs(off_diag))),
        "preview_mean_absolute_upper_triangle": float(
            np.mean(np.abs(preview[np.triu_indices(10, k=1)]))
        ),
    }

    region_frame = pd.DataFrame(
        {
            "region_index": np.arange(1, n_regions + 1, dtype=int),
            "mean": region_means,
            "std": region_stds,
        }
    )

    preview_labels = [f"region_{idx:03d}" for idx in range(1, 11)]
    preview_frame = pd.DataFrame(preview, index=preview_labels, columns=preview_labels)
    return summary, region_frame, preview_frame


def write_outputs(
    summary: dict[str, object],
    region_frame: pd.DataFrame,
    preview_frame: pd.DataFrame,
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    (output_dir / "demo_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    region_frame.to_csv(output_dir / "demo_region_summary.csv", index=False)
    preview_frame.to_csv(output_dir / "demo_fc_preview.csv")

    fig, ax = plt.subplots(figsize=(5.4, 4.8), constrained_layout=True)
    image = ax.imshow(preview_frame.values, cmap="viridis", aspect="equal")
    ax.set_title("Demo FC preview (10 x 10)")
    ax.set_xlabel("Region")
    ax.set_ylabel("Region")
    ax.set_xticks(range(10))
    ax.set_yticks(range(10))
    ax.set_xticklabels(preview_frame.columns, rotation=45, ha="right", fontsize=7)
    ax.set_yticklabels(preview_frame.index, fontsize=7)
    fig.colorbar(image, ax=ax, shrink=0.82, label="Correlation")
    fig.savefig(output_dir / "demo_fc_preview.png", dpi=200)
    plt.close(fig)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path, help="Path to the simulated demo .npy file.")
    parser.add_argument(
        "--output-dir",
        required=True,
        type=Path,
        help="Directory that will receive the demo outputs.",
    )
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    ts = load_timeseries(args.input)
    summary, region_frame, preview_frame = compute_summary(ts)
    summary["input_path"] = str(args.input)
    summary["output_files"] = [
        "demo_summary.json",
        "demo_region_summary.csv",
        "demo_fc_preview.csv",
        "demo_fc_preview.png",
    ]
    write_outputs(summary, region_frame, preview_frame, args.output_dir)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

