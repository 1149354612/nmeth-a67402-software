"""Generate a compact main-text evidence summary figure.

The figure uses current manuscript reporting sources only. It summarizes the
three key evidence roles without changing the underlying data values:
within-subject held-out FC reconstruction, simplified LOSO generalization, and
fixed CCEP correspondence.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
FIGURES_DIR = ROOT / "figures"
OUT_PNG = FIGURES_DIR / "figure5_evidence_summary.png"
OUT_PDF = FIGURES_DIR / "figure5_evidence_summary.pdf"
OUT_META = FIGURES_DIR / "figure5_evidence_summary_metadata.json"

HELDOUT_SOURCE = (
    ROOT
    / "natneurosci_submission"
    / "supplementary_analyses"
    / "fc_correlation_results"
    / "fc_correlation_summary.json"
)
LOSO_SOURCE = (
    ROOT
    / "natneurosci_submission"
    / "supplementary_analyses"
    / "loocv_results"
    / "summary.json"
)
CCEP_SOURCE = ROOT / "reproduce0866_ccep_fixed_pipeline" / "summary_statistics.json"


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Required Figure 5 source file is missing: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _rel(path: Path) -> str:
    return str(path.relative_to(ROOT)).replace("\\", "/")


def load_evidence_records() -> list[dict[str, Any]]:
    heldout = _read_json(HELDOUT_SOURCE)["fc_correlation"]
    loso_payload = _read_json(LOSO_SOURCE)
    loso = loso_payload["fc_correlation"]
    ccep = _read_json(CCEP_SOURCE)["primary_analysis"]

    return [
        {
            "key": "heldout_fc",
            "label": "Within-subject\nheld-out FC",
            "short_label": "Held-out FC",
            "mean_r": float(heldout["mean"]),
            "sd_r": float(heldout["std"]),
            "ci_95": [float(v) for v in heldout["ci_95"]],
            "n": int(_read_json(HELDOUT_SOURCE)["n_subjects"]),
            "unit": "subjects",
            "role": "Primary within-subject reconstruction",
            "limitation": "Small fixed held-out set; subject-specific interpretation.",
            "source": _rel(HELDOUT_SOURCE),
        },
        {
            "key": "simplified_loso_fc",
            "label": "Cross-subject\nLOSO FC",
            "short_label": "LOSO FC",
            "mean_r": float(loso["mean"]),
            "sd_r": float(loso["std"]),
            "ci_95": [float(v) for v in loso["ci_95"]],
            "n": int(loso_payload["n_folds"]),
            "unit": "folds",
            "role": "Simplified cross-subject feasibility",
            "limitation": (
                "Memory-limited LOSO estimate; not final fixed-configuration "
                "cross-subject performance."
            ),
            "source": _rel(LOSO_SOURCE),
        },
        {
            "key": "fixed_ccep",
            "label": "Fixed CCEP\ncomparison",
            "short_label": "CCEP",
            "mean_r": float(ccep["mean_r"]),
            "sd_r": float(ccep["sd_r"]),
            "ci_95": [float(v) for v in ccep["fisher_z_95ci"]],
            "n": int(ccep["n_subjects"]),
            "unit": "held-out model subjects",
            "role": "External correspondence context",
            "limitation": (
                "Fixed post-exploratory comparison with low effect size; "
                "not definitive external validation."
            ),
            "source": _rel(CCEP_SOURCE),
        },
    ]


def _configure_matplotlib() -> None:
    plt.rcParams.update(
        {
            "font.family": "Arial",
            "font.size": 8,
            "axes.titlesize": 9,
            "axes.labelsize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "figure.dpi": 300,
            "savefig.dpi": 300,
        }
    )


def build_figure(records: list[dict[str, Any]]):
    _configure_matplotlib()
    fig, ax_bar = plt.subplots(figsize=(7.1, 1.95), constrained_layout=True)

    means = np.array([record["mean_r"] for record in records], dtype=float)
    lows = np.array([record["ci_95"][0] for record in records], dtype=float)
    highs = np.array([record["ci_95"][1] for record in records], dtype=float)
    y = np.arange(len(records))[::-1]

    colors = ["#285a84", "#7894a8", "#b9853a"]
    edge_colors = ["#183955", "#526878", "#76511d"]
    for idx, (record, ypos) in enumerate(zip(records, y)):
        ax_bar.hlines(ypos, lows[idx], highs[idx], color=colors[idx], linewidth=1.5)
        ax_bar.plot([lows[idx], lows[idx]], [ypos - 0.055, ypos + 0.055], color=colors[idx], linewidth=0.9)
        ax_bar.plot([highs[idx], highs[idx]], [ypos - 0.055, ypos + 0.055], color=colors[idx], linewidth=0.9)
        ax_bar.scatter(
            means[idx],
            ypos,
            s=36,
            color=colors[idx],
            edgecolor=edge_colors[idx],
            linewidth=0.6,
            zorder=3,
        )
        ax_bar.annotate(
            f"r={record['mean_r']:.3f}; n={record['n']}",
            xy=(highs[idx], ypos),
            xytext=(14, 0),
            textcoords="offset points",
            ha="left",
            va="center",
            fontsize=7,
            color="#222222",
            clip_on=False,
        )
        ax_bar.text(
            0.52,
            ypos - 0.30,
            record["role"],
            ha="left",
            va="center",
            fontsize=6.8,
            color="#555555",
        )

    ax_bar.axvline(0, color="#333333", linewidth=0.8)
    ax_bar.set_yticks(y)
    ax_bar.set_yticklabels(
        [
            "Held-out FC\nwithin subject",
            "LOSO FC\ncross subject",
            "CCEP\nfixed comparison",
        ]
    )
    ax_bar.set_xlabel("Correlation (r, mean and 95% CI)")
    ax_bar.set_xlim(-0.02, 1.13)
    ax_bar.set_ylim(-0.55, len(records) - 0.45)
    ax_bar.set_title("Evidence context", loc="left", fontweight="bold", pad=5)
    ax_bar.spines[["top", "right"]].set_visible(False)
    ax_bar.grid(axis="x", color="#e7e7e7", linewidth=0.5)
    ax_bar.set_axisbelow(True)
    fig.text(
        0.995,
        0.03,
        "Summary only; no new analysis.",
        ha="right",
        va="bottom",
        fontsize=6.8,
        color="#666666",
    )

    return fig


def write_figure5(records: list[dict[str, Any]] | None = None, out_dir: Path | None = None) -> dict[str, Path]:
    if records is None:
        records = load_evidence_records()
    if out_dir is None:
        out_dir = FIGURES_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    png = out_dir / "figure5_evidence_summary.png"
    pdf = out_dir / "figure5_evidence_summary.pdf"
    metadata_path = out_dir / "figure5_evidence_summary_metadata.json"

    fig = build_figure(records)
    fig.savefig(png, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)

    metadata = {
        "figure": "Figure 5 evidence summary",
        "style_version": "compact_single_panel_v4",
        "display_role": "contextual_summary",
        "purpose": (
            "Low-prominence main-text context panel summarizing the manuscript's "
            "primary reconstruction result, cross-subject generalization gap, "
            "and fixed CCEP correspondence."
        ),
        "interpretation_boundary": (
            "This figure summarizes current manuscript evidence without adding "
            "new analyses or changing values."
        ),
        "records": records,
        "outputs": {
            "png": _rel(png) if png.is_relative_to(ROOT) else str(png),
            "pdf": _rel(pdf) if pdf.is_relative_to(ROOT) else str(pdf),
        },
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    return {"png": png, "pdf": pdf, "metadata": metadata_path}


def main() -> None:
    outputs = write_figure5()
    print(
        json.dumps(
            {key: _rel(path) if path.is_relative_to(ROOT) else str(path) for key, path in outputs.items()},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
