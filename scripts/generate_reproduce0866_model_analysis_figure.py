"""Generate a data-driven held-out model analysis figure.

This figure uses only fixed held-out reproduce_0866 outputs:
empirical FC, model FC, perturbational EC, and subject-level FC r values.
It deliberately does not plot a training-loss curve because verified loss
history files are not available for the fixed held-out reporting runs.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SUBJECTS = ["106824", "117021", "123723", "136631", "151930", "152225"]
OUT_PNG = ROOT / "figures" / "figure2_heldout_model_analysis_reproduce0866.png"
OUT_PDF = ROOT / "figures" / "figure2_heldout_model_analysis_reproduce0866.pdf"
OUT_META = ROOT / "figures" / "figure2_heldout_model_analysis_reproduce0866_metadata.json"


def offdiag_values(mat: np.ndarray) -> np.ndarray:
    mask = ~np.eye(mat.shape[0], dtype=bool)
    return mat[mask]


def corr(a: np.ndarray, b: np.ndarray) -> float:
    mask = np.isfinite(a) & np.isfinite(b)
    if mask.sum() < 3:
        return float("nan")
    return float(np.corrcoef(a[mask], b[mask])[0, 1])


def load_records() -> list[dict]:
    records = []
    for subject in SUBJECTS:
        run_dir = ROOT / "validation_results" / subject / f"reproduce_0866_{subject}"
        if "reproduce_0866" not in str(run_dir):
            raise RuntimeError(f"Unexpected non-reproduce path: {run_dir}")
        empirical_fc = np.load(run_dir / "empirical_fc.npy")
        model_fc = np.load(run_dir / "model_fc.npy")
        ec = np.load(run_dir / "ec.npy")
        fc_r = float((run_dir / "fc_r.txt").read_text().strip())
        records.append(
            {
                "subject": subject,
                "run_dir": str(run_dir.relative_to(ROOT)),
                "empirical_fc": empirical_fc,
                "model_fc": model_fc,
                "ec": ec,
                "fc_r": fc_r,
                "ec_fc_r": corr(offdiag_values(ec), offdiag_values(empirical_fc)),
            }
        )
    return records


def build_figure(records: list[dict]):
    fc_rs = np.array([r["fc_r"] for r in records], dtype=float)
    mean_fc = float(fc_rs.mean())
    population_sd_fc = float(fc_rs.std(ddof=0))
    sample_sd_fc = float(fc_rs.std(ddof=1))
    rep = min(records, key=lambda r: abs(r["fc_r"] - mean_fc))
    rep_subject = rep["subject"]

    emp_vec = offdiag_values(rep["empirical_fc"])
    mod_vec = offdiag_values(rep["model_fc"])
    ec_vec = offdiag_values(rep["ec"])
    finite_fc = np.isfinite(emp_vec) & np.isfinite(mod_vec)
    finite_ec = np.isfinite(ec_vec)
    rep_fc_corr = corr(emp_vec, mod_vec)

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

    fig = plt.figure(figsize=(7.1, 5.0))
    gs = fig.add_gridspec(2, 3, width_ratios=[1.05, 1.15, 1.15], height_ratios=[1, 1])

    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[1, 0])
    ax_c = fig.add_subplot(gs[0, 1])
    ax_d = fig.add_subplot(gs[0, 2])
    ax_e = fig.add_subplot(gs[1, 1])
    ax_f = fig.add_subplot(gs[1, 2])

    # a, Study split.
    ax_a.axis("off")
    ax_a.set_title("a  Study split and fixed outputs", loc="left", fontweight="bold")
    ax_a.add_patch(plt.Rectangle((0.06, 0.68), 0.88, 0.18, fc="#e9eef5", ec="#33495f", lw=1.0))
    ax_a.text(0.50, 0.77, "33 HCP subjects", ha="center", va="center", fontweight="bold")
    ax_a.annotate("", xy=(0.25, 0.58), xytext=(0.25, 0.68), arrowprops=dict(arrowstyle="-|>", lw=0.8, color="#33495f"))
    ax_a.annotate("", xy=(0.76, 0.58), xytext=(0.76, 0.68), arrowprops=dict(arrowstyle="-|>", lw=0.8, color="#33495f"))
    ax_a.add_patch(plt.Rectangle((0.02, 0.38), 0.45, 0.20, fc="#f4f4f4", ec="#6b6b6b", lw=0.8))
    ax_a.add_patch(plt.Rectangle((0.57, 0.38), 0.39, 0.20, fc="#f8efe1", ec="#8a6f35", lw=0.8))
    ax_a.text(0.245, 0.48, "Development\nn=27", ha="center", va="center", fontsize=7.3)
    ax_a.text(0.765, 0.48, "Held-out\nn=6", ha="center", va="center", fontsize=7.3)
    ax_a.text(
        0.50,
        0.20,
        "Fixed outputs:\nempirical FC, model FC,\nperturbational EC",
        ha="center",
        va="center",
        fontsize=7.0,
        color="#333333",
    )

    # b, Held-out FC reconstruction.
    ax_b.set_title("b  Held-out FC reconstruction", loc="left", fontweight="bold")
    x = np.arange(len(SUBJECTS))
    ax_b.bar(x, fc_rs, color="#6f8fb8", edgecolor="#34495e", linewidth=0.6)
    ax_b.axhline(mean_fc, color="#b3403a", lw=1.2, label=f"mean={mean_fc:.3f}")
    ax_b.set_xticks(x)
    ax_b.set_xticklabels(SUBJECTS, rotation=45, ha="right")
    ax_b.set_ylabel("Pearson r")
    ax_b.set_ylim(0.72, 0.94)
    ax_b.text(0.02, 0.94, f"six-subject mean={mean_fc:.3f}", transform=ax_b.transAxes, va="top")
    ax_b.spines[["top", "right"]].set_visible(False)

    # c-d, FC matrices for representative subject.
    vlim = 1.0
    ax_c.imshow(rep["empirical_fc"], cmap="RdBu_r", vmin=-vlim, vmax=vlim, interpolation="nearest")
    ax_c.set_title(f"c  Empirical FC\nsubject {rep_subject}", loc="left", fontweight="bold")
    ax_c.set_xticks([])
    ax_c.set_yticks([])
    im_d = ax_d.imshow(rep["model_fc"], cmap="RdBu_r", vmin=-vlim, vmax=vlim, interpolation="nearest")
    ax_d.set_title("d  Model FC\nsame subject", loc="left", fontweight="bold")
    ax_d.set_xticks([])
    ax_d.set_yticks([])

    # e, Empirical vs model FC.
    ax_e.set_title("e  Empirical FC vs model FC", loc="left", fontweight="bold")
    ax_e.hexbin(emp_vec[finite_fc], mod_vec[finite_fc], gridsize=70, cmap="Blues", mincnt=1, linewidths=0)
    lim = [-1, 1]
    ax_e.plot(lim, lim, color="#666666", lw=0.7, ls="--")
    ax_e.set_xlim(lim)
    ax_e.set_ylim(lim)
    ax_e.set_xlabel("Empirical FC")
    ax_e.set_ylabel("Model FC")
    ax_e.text(0.04, 0.94, f"r={rep_fc_corr:.3f}", transform=ax_e.transAxes, va="top")
    ax_e.spines[["top", "right"]].set_visible(False)

    # f, EC distribution for representative subject. This avoids introducing
    # a second EC-FC correlation convention outside the fixed comparison script.
    ax_f.set_title("f  Perturbational EC distribution", loc="left", fontweight="bold")
    ec_abs_clip = np.percentile(np.abs(ec_vec[finite_ec]), 99)
    hist_vals = ec_vec[finite_ec]
    hist_vals = hist_vals[np.abs(hist_vals) <= ec_abs_clip]
    ax_f.hist(hist_vals, bins=80, color="#7f6aa8", edgecolor="white", linewidth=0.2)
    ax_f.axvline(0, color="#555555", lw=0.8)
    ax_f.set_xlim(-ec_abs_clip, ec_abs_clip)
    ax_f.set_xlabel("Model-implied EC")
    ax_f.set_ylabel("Off-diagonal edge count")
    ax_f.text(0.04, 0.94, "99% central range", transform=ax_f.transAxes, va="top")
    ax_f.spines[["top", "right"]].set_visible(False)

    fig.tight_layout(rect=[0, 0, 0.955, 1.0])
    fig.canvas.draw()
    d_pos = ax_d.get_position()
    cax = fig.add_axes([d_pos.x1 + 0.018, d_pos.y0, 0.012, d_pos.height])
    cb = fig.colorbar(im_d, cax=cax)
    cb.set_label("FC")

    metadata = {
        "purpose": "Data-driven held-out model analysis figure for manuscript revision.",
        "source_rule": "Only validation_results/{subject}/reproduce_0866_{subject}/ fixed held-out outputs are used.",
        "subjects": SUBJECTS,
        "run_dirs": [r["run_dir"] for r in records],
        "fc_r_by_subject": {r["subject"]: r["fc_r"] for r in records},
        "heldout_fc_mean": mean_fc,
        "heldout_fc_population_sd_matches_summary_json": population_sd_fc,
        "heldout_fc_sample_sd_from_six_values": sample_sd_fc,
        "sd_reporting_note": (
            "The existing manuscript summary reports 0.042, which matches the population "
            "standard deviation of the six fc_r.txt values. The sample SD is 0.046; "
            "the figure labels only the six-subject mean to avoid adding a new SD convention."
        ),
        "representative_subject_rule": "subject whose FC reconstruction r is closest to the six-subject held-out mean",
        "representative_subject": rep_subject,
        "representative_subject_fc_r_file": rep["fc_r"],
        "representative_subject_recomputed_empirical_vs_model_fc_r": rep_fc_corr,
        "representative_subject_ec_distribution_panel": (
            "Panel f shows the central 99% of off-diagonal EC values for the representative subject. "
            "It intentionally does not report EC-FC correlation because the manuscript correlation "
            "uses the fixed reproduce0866_fc_pc_comparison pipeline."
        ),
        "interpretation_boundary": (
            "This is a descriptive model-analysis figure. It is not a training-loss curve, "
            "not external validation, and not biological ground truth."
        ),
        "outputs": {
            "png": str(OUT_PNG.relative_to(ROOT)),
            "pdf": str(OUT_PDF.relative_to(ROOT)),
        },
    }
    axes = {
        "a": ax_a,
        "b": ax_b,
        "c": ax_c,
        "d": ax_d,
        "e": ax_e,
        "f": ax_f,
        "fc_colorbar": cb.ax,
    }
    return fig, axes, metadata


def main() -> None:
    records = load_records()

    fig, _, metadata = build_figure(records)
    OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PNG, bbox_inches="tight")
    fig.savefig(OUT_PDF, bbox_inches="tight")
    plt.close(fig)
    OUT_META.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
