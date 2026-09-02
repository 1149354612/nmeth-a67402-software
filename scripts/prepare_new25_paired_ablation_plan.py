#!/usr/bin/env python
"""Prepare a paired ablation plan for the traceable 25-subject HCP-YA batch.

The plan is intentionally a planning artifact, not a result. It reuses the
completed full MGKA H=60 FC-loss=0.5 run as the reference condition and defines
three matched ablation cells per subject: dot attention, FC-loss removal, and
shorter prediction horizon. Results should not be reported until every
subject-condition cell completes and paired statistics are generated.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parent
TRAIN_SCRIPT = ROOT / "scripts" / "core" / "run_transformer_param.py"
DEFAULT_SUMMARY = ROOT / "prespecified_hcp_evaluation_new25_full_20260702" / "new25_full_summary_20260703.json"
DEFAULT_DATA_DIR = ROOT / "prespecified_hcp_360_inputs"
DEFAULT_PROVENANCE_DIR = ROOT / "prespecified_hcp_provenance"
DEFAULT_FULL_OUTPUT_BASE = ROOT / "prespecified_hcp_evaluation_new25_full_20260702"
DEFAULT_OUTPUT_BASE = ROOT / "new25_paired_ablation_plan"
SEED = 20251211


BASE_CONFIG = {
    "steps": 10,
    "horizon": 60,
    "fc_loss_weight": 0.5,
    "fc_horizon": 60,
    "fc_roi_sample": 0,
    "sign_loss_weight": 0.0,
    "sign_loss_margin": 0.8,
    "sign_loss_min_amp": 0.05,
    "ms_weight_policy": "power",
    "ms_weight_power": 1.5,
    "d_model": 256,
    "n_heads": 4,
    "n_layers": 4,
    "batch_size": 64,
    "total_epochs": 150,
    "warmup_epochs": 10,
    "patience": 15,
    "tf_anneal_epochs": 60,
    "tf_min": 0.1,
    "noise_std": 0.05,
    "gen_steps": 1200,
    "fc_eval_seeds": 16,
    "attn": "mgka",
}


CONFIGS = [
    {
        "name": "full_mgka_h60_fc05",
        "cell_type": "existing_reference",
        "changed_parameter": "none",
        "description": "Completed full MGKA reference condition from the fixed-list traceable 25-subject evaluation.",
        "overrides": {},
    },
    {
        "name": "dot_h60_fc05",
        "cell_type": "train_ablation",
        "changed_parameter": "attention",
        "description": "Dot-product attention with all other parameters matched to the full reference condition.",
        "overrides": {"attn": "dot"},
    },
    {
        "name": "mgka_h60_fc00",
        "cell_type": "train_ablation",
        "changed_parameter": "fc_loss_weight",
        "description": "MGKA with FC consistency loss removed; all other parameters matched.",
        "overrides": {"fc_loss_weight": 0.0},
    },
    {
        "name": "mgka_h30_fc05",
        "cell_type": "train_ablation",
        "changed_parameter": "prediction_horizon",
        "description": "MGKA with shorter prediction horizon and matched FC horizon.",
        "overrides": {"horizon": 30, "fc_horizon": 30},
    },
]


def load_subjects(summary_path: Path = DEFAULT_SUMMARY) -> list[str]:
    summary = json.loads(Path(summary_path).read_text(encoding="utf-8"))
    subjects = [str(subject) for subject in summary.get("subjects", [])]
    if not subjects:
        raise ValueError(f"No subjects listed in {summary_path}")
    if summary.get("n_success") != len(subjects) or summary.get("n_complete") != len(subjects):
        raise ValueError("The source summary must contain only completed successful subjects.")
    return subjects


def model_input_for(subject: str, data_dir: Path) -> Path:
    return Path(data_dir) / f"sub-{subject}_bold_360.npy"


def provenance_for(subject: str, provenance_dir: Path) -> Path:
    return Path(provenance_dir) / f"{subject}_provenance.json"


def full_reference_dir(subject: str, full_output_base: Path) -> Path:
    return Path(full_output_base) / subject / f"prespecified_traceable_hcp_{subject}"


def expected_outputs(output_dir: Path) -> list[str]:
    return [
        str(output_dir / "fc_r.txt"),
        str(output_dir / "config.json"),
        str(output_dir / "ec.npy"),
        str(output_dir / "training_history.csv"),
        str(output_dir / "training_summary.json"),
    ]


def command_for(subject: str, config: dict, data_dir: Path, output_base: Path) -> tuple[list[str], str, Path]:
    args = dict(BASE_CONFIG)
    args.update(config["overrides"])
    run_tag = f"new25_ablation_{config['name']}_{subject}_s{SEED}"
    output_dir = Path(output_base) / subject / run_tag

    command = [
        sys.executable,
        str(TRAIN_SCRIPT),
        "--data_path",
        str(model_input_for(subject, data_dir)),
        "--seed",
        str(SEED),
        "--run_tag",
        run_tag,
        "--output_base_dir",
        str(Path(output_base).resolve()),
    ]

    for key, value in args.items():
        command.extend([f"--{key}", str(value)])

    command.extend(["--use_delta", "--per_run_zscore"])
    return command, run_tag, output_dir


def build_reference_cell(subject: str, full_output_base: Path, data_dir: Path, provenance_dir: Path) -> dict:
    output_dir = full_reference_dir(subject, full_output_base)
    return {
        "subject": subject,
        "config": "full_mgka_h60_fc05",
        "cell_type": "existing_reference",
        "changed_parameter": "none",
        "seed": SEED,
        "data_path": str(model_input_for(subject, data_dir)),
        "provenance_path": str(provenance_for(subject, provenance_dir)),
        "expected_output_dir": str(output_dir),
        "expected_key_outputs": expected_outputs(output_dir),
        "command": [],
        "interpretation": "existing full-model reference cell; paired result requires all ablation cells",
    }


def build_training_cell(
    subject: str,
    config: dict,
    data_dir: Path,
    provenance_dir: Path,
    output_base: Path,
) -> dict:
    command, run_tag, output_dir = command_for(subject, config, data_dir, output_base)
    return {
        "subject": subject,
        "config": config["name"],
        "cell_type": "train_ablation",
        "changed_parameter": config["changed_parameter"],
        "description": config["description"],
        "seed": SEED,
        "run_tag": run_tag,
        "data_path": str(model_input_for(subject, data_dir)),
        "provenance_path": str(provenance_for(subject, provenance_dir)),
        "expected_output_dir": str(output_dir),
        "expected_key_outputs": expected_outputs(output_dir),
        "command": command,
        "command_string": " ".join(command),
        "interpretation": "paired ablation cell; not a manuscript result until all cells complete",
    }


def build_plan(
    subjects: list[str],
    data_dir: Path = DEFAULT_DATA_DIR,
    provenance_dir: Path = DEFAULT_PROVENANCE_DIR,
    full_output_base: Path = DEFAULT_FULL_OUTPUT_BASE,
    output_base: Path = DEFAULT_OUTPUT_BASE,
) -> dict:
    subjects = [str(subject) for subject in subjects]
    data_dir = Path(data_dir)
    provenance_dir = Path(provenance_dir)
    full_output_base = Path(full_output_base)
    output_base = Path(output_base)

    cells = []
    for subject in subjects:
        cells.append(build_reference_cell(subject, full_output_base, data_dir, provenance_dir))
        for config in CONFIGS:
            if config["cell_type"] == "train_ablation":
                cells.append(build_training_cell(subject, config, data_dir, provenance_dir, output_base))

    return {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": "Strictly paired component ablation for the fixed-list traceable 25-subject HCP-YA evaluation.",
        "interpretation_boundary": (
            "This is a planned ablation design and is not reported until all prespecified "
            "subject-condition cells complete and paired statistics are computed."
        ),
        "subjects": subjects,
        "n_subjects": len(subjects),
        "seed": SEED,
        "reference_condition_source": str(full_output_base),
        "output_base": str(output_base),
        "training_script": str(TRAIN_SCRIPT),
        "base_config": BASE_CONFIG,
        "configs": CONFIGS,
        "cells": cells,
        "n_reference_cells": sum(1 for row in cells if row["cell_type"] == "existing_reference"),
        "n_train_ablation_cells": sum(1 for row in cells if row["cell_type"] == "train_ablation"),
        "paired_design": (
            "Each subject has the completed full MGKA reference condition and will be evaluated "
            "under dot attention, FC-loss removal, and shorter-horizon conditions using the same "
            "seed, data file, training script, and evaluation outputs."
        ),
    }


def write_plan(plan: dict, output_base: Path = DEFAULT_OUTPUT_BASE) -> dict[str, Path]:
    output_base = Path(output_base)
    output_base.mkdir(parents=True, exist_ok=True)

    plan_json = output_base / "new25_paired_ablation_plan.json"
    commands_txt = output_base / "commands.txt"

    plan_json.write_text(json.dumps(plan, indent=2), encoding="utf-8")
    commands = [row["command_string"] for row in plan["cells"] if row["cell_type"] == "train_ablation"]
    commands_txt.write_text("\n".join(commands) + ("\n" if commands else ""), encoding="utf-8")

    return {"plan_json": plan_json, "commands_txt": commands_txt}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare the new-25 paired ablation plan.")
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--subjects", nargs="*", default=None)
    parser.add_argument("--limit-subjects", type=int, default=None)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--provenance-dir", type=Path, default=DEFAULT_PROVENANCE_DIR)
    parser.add_argument("--full-output-base", type=Path, default=DEFAULT_FULL_OUTPUT_BASE)
    parser.add_argument("--output-base", type=Path, default=DEFAULT_OUTPUT_BASE)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    subjects = [str(subject) for subject in args.subjects] if args.subjects else load_subjects(args.summary)
    if args.limit_subjects is not None:
        subjects = subjects[: args.limit_subjects]

    plan = build_plan(
        subjects=subjects,
        data_dir=args.data_dir,
        provenance_dir=args.provenance_dir,
        full_output_base=args.full_output_base,
        output_base=args.output_base,
    )
    paths = write_plan(plan, output_base=args.output_base)
    print(
        json.dumps(
            {
                "plan_json": str(paths["plan_json"]),
                "commands_txt": str(paths["commands_txt"]),
                "n_subjects": plan["n_subjects"],
                "n_reference_cells": plan["n_reference_cells"],
                "n_train_ablation_cells": plan["n_train_ablation_cells"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
