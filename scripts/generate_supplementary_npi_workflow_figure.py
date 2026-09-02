"""Generate a publication-grade schematic Supplementary Fig. S2 NPI workflow.

The figure is assembled entirely from Matplotlib primitives and deterministic
schematic arrays. It is not an empirical EC result figure: no subject-level
empirical EC result arrays are loaded or plotted. The schematic matrix exists
only to explain how columns are perturbed source regions i and rows are target
response regions j.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import patches
from matplotlib.colors import LinearSegmentedColormap


ROOT = Path(__file__).resolve().parent
FIGURES = ROOT / "figures"
OUT_PNG = FIGURES / "figure4_npi_workflow_publication.png"
OUT_PDF = FIGURES / "figure4_npi_workflow_publication.pdf"
OUT_META = FIGURES / "figure4_npi_workflow_publication_metadata.json"


COLORS = {
    "ink": "#20252b",
    "muted": "#66737c",
    "light_text": "#7b878f",
    "line": "#b9c3ca",
    "panel": "#f8fafb",
    "panel_edge": "#d8e0e5",
    "blue": "#2b6cb0",
    "blue_light": "#e9f2fb",
    "teal": "#16827f",
    "teal_light": "#e8f4f3",
    "red": "#c94c4c",
    "red_light": "#fbebeb",
    "green": "#3c7d54",
    "green_light": "#edf6ef",
    "gold": "#b7791f",
    "gold_light": "#fff7e8",
    "violet": "#5f5aa2",
    "violet_light": "#f1eff8",
}


def clean_axis(ax) -> None:
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_frame_on(False)


def panel(ax, label: str, title: str) -> None:
    clean_axis(ax)
    ax.text(0.00, 1.02, label, ha="left", va="bottom", fontsize=13, fontweight="bold", color=COLORS["ink"])
    ax.text(0.07, 1.02, title, ha="left", va="bottom", fontsize=10.5, fontweight="bold", color=COLORS["ink"])


def rounded_rect(ax, xy, w, h, face, edge=COLORS["panel_edge"], lw=1.0, radius=0.035, z=1):
    rect = patches.FancyBboxPatch(
        xy,
        w,
        h,
        boxstyle=f"round,pad=0.012,rounding_size={radius}",
        facecolor=face,
        edgecolor=edge,
        linewidth=lw,
        zorder=z,
    )
    ax.add_patch(rect)
    return rect


def arrow(ax, start, end, color=COLORS["muted"], lw=1.15, scale=12):
    ax.add_patch(
        patches.FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=scale,
            linewidth=lw,
            color=color,
            shrinkA=0,
            shrinkB=0,
        )
    )


def draw_input_timeseries(ax) -> None:
    panel(ax, "a", "Resting-state input")
    rounded_rect(ax, (0.06, 0.15), 0.88, 0.66, COLORS["blue_light"], "#9ab9d9")

    rng = np.random.default_rng(20260707)
    x = np.linspace(0.13, 0.87, 80)
    for k in range(5):
        phase = 0.65 * k
        y = 0.28 + k * 0.095 + 0.025 * np.sin(np.linspace(0, 3.5 * np.pi, 80) + phase)
        y += rng.normal(0, 0.007, size=x.size)
        color = "#7a8da4" if k != 2 else COLORS["blue"]
        ax.plot(x, y, color=color, linewidth=1.2 if k == 2 else 0.85, alpha=0.95)

    ax.text(0.50, 0.72, "Regional fMRI time series", ha="center", va="center", fontsize=8.5, color=COLORS["ink"])
    ax.text(0.50, 0.11, "Input window: 10 time steps (~7.2 s)", ha="center", va="top", fontsize=8, color=COLORS["muted"])
    ax.text(0.14, 0.19, "T x 360", ha="left", va="bottom", fontsize=8, color=COLORS["muted"])


def draw_surrogate(ax) -> None:
    panel(ax, "b", "Subject-specific surrogate")
    rounded_rect(ax, (0.05, 0.20), 0.90, 0.58, COLORS["teal_light"], "#9ec9c6")
    ax.text(0.50, 0.70, "Trained Transformer surrogate", ha="center", fontsize=8.8, fontweight="bold", color=COLORS["ink"])

    xs = [0.16, 0.34, 0.52, 0.70]
    for idx, x in enumerate(xs, start=1):
        rounded_rect(ax, (x - 0.065, 0.38), 0.13, 0.16, "white", "#82bdb9", lw=0.9, radius=0.018)
        ax.text(x, 0.49, f"Block {idx}", ha="center", fontsize=6.8, color=COLORS["muted"])
        ax.text(x, 0.425, "MGKA", ha="center", fontsize=7.2, fontweight="bold", color=COLORS["teal"])
        if idx < len(xs):
            arrow(ax, (x + 0.075, 0.46), (xs[idx] - 0.075, 0.46), color="#8aa4a2", lw=0.9, scale=8)

    ax.text(0.50, 0.27, "Learns individual resting dynamics", ha="center", fontsize=8, color=COLORS["muted"])


def draw_virtual_perturbation(ax) -> None:
    panel(ax, "c", "Virtual perturbation")
    rounded_rect(ax, (0.08, 0.16), 0.84, 0.64, COLORS["red_light"], "#e0a5a5")

    center = np.array([0.48, 0.56])
    angles = np.linspace(0, 2 * np.pi, 12, endpoint=False)
    radius = 0.18
    pts = np.column_stack([center[0] + radius * np.cos(angles), center[1] + radius * np.sin(angles)])
    for p in pts:
        ax.add_patch(patches.Circle(p, 0.026, facecolor="#9fb6d9", edgecolor="#6481a8", linewidth=0.8))
    src = pts[1]
    ax.add_patch(patches.Circle(src, 0.043, facecolor=COLORS["red"], edgecolor="#8f2f2f", linewidth=1.0, zorder=4))
    ax.text(0.75, 0.63, "source i", fontsize=8.2, color=COLORS["red"], va="center")
    ax.text(0.50, 0.29, "Set source region i to +1", ha="center", fontsize=8.2, color=COLORS["ink"])
    ax.text(0.50, 0.22, "Other regions remain at baseline", ha="center", fontsize=7.6, color=COLORS["muted"])


def draw_response_profile(ax) -> None:
    panel(ax, "d", "Response measurement")
    rounded_rect(ax, (0.07, 0.16), 0.86, 0.64, COLORS["green_light"], "#a4c7ae")
    xs = np.array([0.22, 0.38, 0.54, 0.70])
    heights = np.array([0.24, -0.10, 0.17, -0.05])
    base = 0.47
    for x, h in zip(xs, heights):
        color = COLORS["red"] if h > 0 else COLORS["blue"]
        y0 = base if h > 0 else base + h
        ax.add_patch(patches.Rectangle((x - 0.035, y0), 0.07, abs(h), facecolor=color, alpha=0.72, edgecolor="none"))
        ax.text(x, 0.22, "j", ha="center", fontsize=7.5, color=COLORS["muted"])
    ax.axhline(base, xmin=0.16, xmax=0.78, color="#82908f", linewidth=0.9)
    ax.text(0.50, 0.71, "Steady-state responses", ha="center", fontsize=8.7, fontweight="bold", color=COLORS["ink"])
    ax.text(0.50, 0.16, "Average target-region changes", ha="center", fontsize=7.4, color=COLORS["muted"])
    ax.text(0.50, 0.105, "over virtual windows", ha="center", fontsize=7.2, color=COLORS["muted"])


def draw_ec_rule(ax) -> None:
    panel(ax, "e", "EC assembly rule")
    rounded_rect(ax, (0.05, 0.13), 0.90, 0.70, COLORS["gold_light"], "#d8bb7c")

    ax.text(0.50, 0.70, "For each perturbed source column i", ha="center", fontsize=9, fontweight="bold", color=COLORS["ink"])
    ax.text(0.50, 0.58, "record target row j response", ha="center", fontsize=8.8, color=COLORS["ink"])
    ax.text(0.50, 0.43, r"$EC_{ji}$ = response of target j", ha="center", fontsize=10, color=COLORS["ink"])
    ax.text(0.50, 0.33, "to perturbation at source i", ha="center", fontsize=9.2, color=COLORS["ink"])
    ax.text(0.50, 0.18, "Direction convention: source column i -> target row j", ha="center", fontsize=7.8, color=COLORS["muted"])


def schematic_matrix(n=24) -> np.ndarray:
    x = np.linspace(-1, 1, n)
    xx, yy = np.meshgrid(x, x)
    mat = 0.45 * np.sin(2.5 * np.pi * xx) * np.cos(1.7 * np.pi * yy)
    mat += 0.18 * np.sin(5 * (xx + yy))
    mat -= np.diag(np.diag(mat))
    mat[:, 8] += np.linspace(0.45, -0.35, n)
    mat[14, :] += 0.25 * np.cos(np.linspace(0, 2 * np.pi, n))
    return np.clip(mat, -1, 1)


def draw_output_matrix(ax) -> None:
    panel(ax, "f", "Directed EC output")
    rounded_rect(ax, (0.03, 0.07), 0.94, 0.82, COLORS["panel"], COLORS["panel_edge"])

    inset = ax.inset_axes([0.13, 0.24, 0.52, 0.52])
    cmap = LinearSegmentedColormap.from_list("ec_schematic", ["#2b6cb0", "#f8f8f5", "#c94c4c"])
    im = inset.imshow(schematic_matrix(), cmap=cmap, vmin=-1, vmax=1, interpolation="nearest")
    inset.set_xticks([0, 8, 23])
    inset.set_yticks([0, 14, 23])
    inset.set_xticklabels(["1", "i", "360"], fontsize=7)
    inset.set_yticklabels(["1", "j", "360"], fontsize=7)
    inset.set_xlabel("")
    inset.set_ylabel("target row j", fontsize=7.5)
    inset.tick_params(length=2)
    inset.spines[:].set_linewidth(0.6)
    inset.axvline(8, color="#111111", linewidth=1.0, alpha=0.75)
    inset.axhline(14, color="#111111", linewidth=1.0, alpha=0.75)
    inset.plot(8, 14, marker="s", markersize=5, color="#111111")

    cax = ax.inset_axes([0.72, 0.28, 0.033, 0.42])
    cb = plt.colorbar(im, cax=cax)
    cb.set_label("schematic EC", fontsize=7.2)
    cb.ax.tick_params(labelsize=7, length=2)

    ax.text(0.39, 0.82, "360 x 360 response map", ha="center", fontsize=9.2, fontweight="bold", color=COLORS["ink"])
    ax.text(0.39, 0.155, "source column i", ha="center", fontsize=7.4, color=COLORS["ink"])
    ax.text(0.39, 0.085, "Schematic matrix; no empirical EC result is shown", ha="center", fontsize=7.4, color=COLORS["muted"])


def build_figure():
    FIGURES.mkdir(exist_ok=True)
    fig = plt.figure(figsize=(13.2, 6.9), dpi=300)
    gs = fig.add_gridspec(2, 4, height_ratios=[1.0, 1.18], hspace=0.38, wspace=0.22)
    axes = [
        fig.add_subplot(gs[0, 0]),
        fig.add_subplot(gs[0, 1]),
        fig.add_subplot(gs[0, 2]),
        fig.add_subplot(gs[0, 3]),
        fig.add_subplot(gs[1, 0:2]),
        fig.add_subplot(gs[1, 2:4]),
    ]

    draw_input_timeseries(axes[0])
    draw_surrogate(axes[1])
    draw_virtual_perturbation(axes[2])
    draw_response_profile(axes[3])
    draw_ec_rule(axes[4])
    draw_output_matrix(axes[5])

    # Visual data-flow arrows across the top row.
    for left, right in zip(axes[:3], axes[1:4]):
        p0 = left.transAxes.transform((0.96, 0.48))
        p1 = right.transAxes.transform((0.04, 0.48))
        inv = fig.transFigure.inverted()
        start = inv.transform(p0)
        end = inv.transform(p1)
        fig.patches.append(
            patches.FancyArrowPatch(
                start,
                end,
                transform=fig.transFigure,
                arrowstyle="-|>",
                mutation_scale=11,
                linewidth=1.05,
                color=COLORS["muted"],
            )
        )

    fig.text(0.5, 0.965, "Neural perturbational inference workflow", ha="center", va="top", fontsize=13.2, fontweight="bold", color=COLORS["ink"])
    fig.text(0.5, 0.932, "Code-generated schematic; no empirical EC result panel is shown", ha="center", va="top", fontsize=8.4, color=COLORS["muted"])

    fig.savefig(OUT_PNG, dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(OUT_PDF, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    metadata = {
        "figure": "Supplementary Fig. S2 NPI workflow",
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "generation_mode": "matplotlib_code_only",
        "uses_ai_generated_components": False,
        "contains_empirical_result_panels": False,
        "contains_schematic_matrix_only": True,
        "ec_direction_convention": "columns are perturbed source regions i; rows are target response regions j",
        "source_data_boundary": "No subject-level empirical EC result arrays are plotted in this schematic.",
        "outputs": {
            "png": str(OUT_PNG.relative_to(ROOT)),
            "pdf": str(OUT_PDF.relative_to(ROOT)),
        },
        "notes": [
            "The figure explains the NPI workflow and EC assembly convention.",
            "The matrix, response bars, and network nodes are deterministic schematic graphics.",
            "No HCP subject-level EC result arrays are read by this script.",
        ],
    }
    OUT_META.write_text(json.dumps(metadata, indent=2), encoding="utf-8")


if __name__ == "__main__":
    build_figure()
