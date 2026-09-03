"""Run a prespecified, traceable HCP-derived fixed-configuration evaluation.

This script freezes a new candidate subject set before model outcomes are
computed. It deliberately fails closed: a subject is eligible for training only
when a 360-column model input exists and a per-subject provenance JSON records
the model-input hash, source files, and processing scripts.

The current repository does not contain the raw HCP inputs for this new batch.
In that state the script writes an audit/status file with missing-input records
rather than fabricating a completed experiment.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent
for import_root in (ROOT, SCRIPT_DIR):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

import numpy as np

import run_reproduce0866_training_diagnostics as heldout_diag


DATA_DIR = ROOT / "NPI-main" / "NPI-main" / "real_fMRI_data"
DEFAULT_PROVENANCE_DIR = ROOT / "prespecified_hcp_provenance"
DEFAULT_OUTPUT_BASE = ROOT / "prespecified_hcp_evaluation"
STATUS_FILENAME = "prespecified_hcp_evaluation_status.json"
COHORT_LABEL = "prespecified_traceable_hcp_20260701"
ALLOWED_PARCELLATION_METHODS = {
    "official_hcp_mmp1_360",
    "existing_360_ptseries",
    "four_run_360_txt_merge",
}

CURRENT_53_SUBJECTS = [
    "102008",
    "102109",
    "102614",
    "102715",
    "103212",
    "106824",
    "108020",
    "111211",
    "113316",
    "115724",
    "117021",
    "118831",
    "119025",
    "120414",
    "123723",
    "130518",
    "136631",
    "137532",
    "138130",
    "138332",
    "139435",
    "143224",
    "145632",
    "146735",
    "146836",
    "147636",
    "151324",
    "151930",
    "152225",
    "152427",
    "153126",
    "161832",
    "165436",
    "175136",
    "176845",
    "177140",
    "180230",
    "188145",
    "191235",
    "192237",
    "193845",
    "194443",
    "198047",
    "199352",
    "200513",
    "206323",
    "206525",
    "206727",
    "206828",
    "206929",
    "210112",
    "211619",
    "211821",
]

PRESPECIFIED_SUBJECTS = [
    "102311",
    "102513",
    "102816",
    "103010",
    "103111",
    "103414",
    "103515",
    "103818",
    "104012",
    "104416",
    "104820",
    "105014",
    "105115",
    "105216",
    "105923",
    "106016",
    "106319",
    "106521",
    "107018",
    "107321",
    "107422",
    "108121",
    "108323",
    "108525",
    "108828",
]

REQUIRED_OUTPUTS = heldout_diag.REQUIRED_OUTPUTS
COMPLETE_STATUSES = heldout_diag.COMPLETE_STATUSES
STATUS_RANK = {
    **heldout_diag.STATUS_RANK,
    "excluded_existing_subject": -2,
    "missing_model_input": -1,
    "missing_provenance": -1,
    "invalid_model_input": -1,
    "invalid_provenance": -1,
    "eligible": 0,
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def data_path_for_subject(subject: str, data_dir: Path = DATA_DIR) -> Path:
    return Path(data_dir) / f"sub-{subject}_bold_360.npy"


def provenance_path_for_subject(subject: str, provenance_dir: Path = DEFAULT_PROVENANCE_DIR) -> Path:
    return Path(provenance_dir) / f"{subject}_provenance.json"


def _resolve_recorded_path(path_text: str) -> Path:
    path = Path(path_text)
    if path.is_absolute():
        return path
    return ROOT / path


def _base_record(subject: str, data_dir: Path, provenance_dir: Path) -> dict:
    return {
        "subject": str(subject),
        "cohort": COHORT_LABEL,
        "eligible_for_training": False,
        "expected_model_input": str(data_path_for_subject(subject, data_dir)),
        "expected_provenance": str(provenance_path_for_subject(subject, provenance_dir)),
    }


def audit_subject_input(
    subject: str,
    data_dir: Path = DATA_DIR,
    provenance_dir: Path = DEFAULT_PROVENANCE_DIR,
) -> dict:
    subject = str(subject)
    data_dir = Path(data_dir)
    provenance_dir = Path(provenance_dir)
    record = _base_record(subject, data_dir, provenance_dir)

    if subject in set(CURRENT_53_SUBJECTS):
        record["status"] = "excluded_existing_subject"
        record["reason"] = "Subject already belongs to the current 33+20 processed-input set."
        return record

    model_input = data_path_for_subject(subject, data_dir)
    if not model_input.exists():
        record["status"] = "missing_model_input"
        record["reason"] = "No prespecified 360-column model input is available locally."
        return record

    try:
        array = np.load(str(model_input))
    except Exception as exc:
        record["status"] = "invalid_model_input"
        record["reason"] = f"Could not load model input: {exc}"
        return record

    record["shape"] = [int(array.shape[0]), int(array.shape[1])] if array.ndim == 2 else list(array.shape)
    if array.ndim != 2 or array.shape[1] != 360:
        record["status"] = "invalid_model_input"
        record["reason"] = "Model input must be a 2D array with exactly 360 columns."
        return record
    if array.shape[0] <= 70:
        record["status"] = "invalid_model_input"
        record["reason"] = "Model input has too few timepoints for steps=10 and horizon=60."
        return record
    if not np.isfinite(array).all():
        record["status"] = "invalid_model_input"
        record["reason"] = "Model input contains NaN or Inf values."
        return record

    model_hash = sha256_file(model_input)
    record["model_input_sha256"] = model_hash

    provenance_path = provenance_path_for_subject(subject, provenance_dir)
    if not provenance_path.exists():
        record["status"] = "missing_provenance"
        record["reason"] = "A per-subject provenance JSON is required before training."
        return record

    try:
        provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    except Exception as exc:
        record["status"] = "invalid_provenance"
        record["reason"] = f"Could not parse provenance JSON: {exc}"
        return record

    if str(provenance.get("subject")) != subject:
        record["status"] = "invalid_provenance"
        record["reason"] = "Provenance subject does not match the audited subject."
        return record
    if provenance.get("model_input_sha256") != model_hash:
        record["status"] = "invalid_provenance"
        record["reason"] = "Provenance model_input_sha256 does not match the current file."
        return record
    source_files = provenance.get("source_files")
    if not isinstance(source_files, list) or not source_files:
        record["status"] = "invalid_provenance"
        record["reason"] = "Provenance must list at least one source file."
        return record
    missing_sources = [
        str(_resolve_recorded_path(str(source_file)))
        for source_file in source_files
        if not _resolve_recorded_path(str(source_file)).exists()
    ]
    if missing_sources:
        record["status"] = "invalid_provenance"
        record["reason"] = "One or more source files recorded in provenance are missing."
        record["missing_source_files"] = missing_sources
        return record
    scripts = provenance.get("processing_scripts")
    if not isinstance(scripts, list) or not scripts:
        record["status"] = "invalid_provenance"
        record["reason"] = "Provenance must list processing scripts."
        return record
    parcellation_method = provenance.get("parcellation_method")
    if parcellation_method not in ALLOWED_PARCELLATION_METHODS:
        record["status"] = "invalid_provenance"
        record["reason"] = (
            "Provenance parcellation_method must be one of "
            f"{sorted(ALLOWED_PARCELLATION_METHODS)}."
        )
        return record

    record["status"] = "eligible"
    record["eligible_for_training"] = True
    record["source_files"] = source_files
    record["processing_scripts"] = scripts
    record["parcellation_method"] = parcellation_method
    return record


def build_prespecified_fixed_config(subject: str, data_path: Path | None = None) -> dict:
    path = Path(data_path) if data_path is not None else data_path_for_subject(subject)
    return {
        "seed": 20251211,
        "batch_size": 64,
        "fc_roi_sample": 0,
        "fc_amp": False,
        "steps": 10,
        "horizon": 60,
        "d_model": 256,
        "n_heads": 4,
        "n_layers": 4,
        "fc_loss_weight": 0.5,
        "fc_horizon": 60,
        "ms_weight_policy": "power",
        "ms_weight_power": 1.5,
        "sign_loss_weight": 0.0,
        "sign_loss_margin": 0.8,
        "sign_loss_min_amp": 0.05,
        "total_epochs": 150,
        "warmup_epochs": 10,
        "patience": 15,
        "tf_anneal_epochs": 60,
        "tf_min": 0.1,
        "noise_std": 0.05,
        "gen_steps": 1200,
        "use_delta": True,
        "per_run_zscore": True,
        "fc_eval_seeds": 16,
        "attn": "mgka",
        "data_path": str(path),
    }


def _add_flag(command: list[str], name: str, value) -> None:
    command.extend([name, str(value)])


def build_training_command(subject: str, config: dict, output_base: Path) -> tuple[list[str], str]:
    data_path = Path(config["data_path"])
    if not data_path.exists():
        raise FileNotFoundError(f"Missing data for prespecified subject {subject}: {data_path}")
    run_tag = f"prespecified_traceable_hcp_{subject}"
    command = [
        sys.executable,
        str(ROOT / "scripts" / "core" / "run_transformer_param.py"),
    ]
    for flag, key in [
        ("--data_path", "data_path"),
        ("--seed", "seed"),
        ("--steps", "steps"),
        ("--horizon", "horizon"),
        ("--fc_loss_weight", "fc_loss_weight"),
        ("--fc_horizon", "fc_horizon"),
        ("--fc_roi_sample", "fc_roi_sample"),
        ("--sign_loss_weight", "sign_loss_weight"),
        ("--sign_loss_margin", "sign_loss_margin"),
        ("--sign_loss_min_amp", "sign_loss_min_amp"),
        ("--ms_weight_policy", "ms_weight_policy"),
        ("--ms_weight_power", "ms_weight_power"),
        ("--d_model", "d_model"),
        ("--n_heads", "n_heads"),
        ("--n_layers", "n_layers"),
        ("--batch_size", "batch_size"),
        ("--total_epochs", "total_epochs"),
        ("--warmup_epochs", "warmup_epochs"),
        ("--patience", "patience"),
        ("--tf_anneal_epochs", "tf_anneal_epochs"),
        ("--tf_min", "tf_min"),
        ("--noise_std", "noise_std"),
        ("--gen_steps", "gen_steps"),
        ("--fc_eval_seeds", "fc_eval_seeds"),
        ("--attn", "attn"),
    ]:
        _add_flag(command, flag, config[key])
    if bool(config.get("fc_amp", False)):
        command.append("--fc_amp")
    if bool(config.get("use_delta", True)):
        command.append("--use_delta")
    if bool(config.get("per_run_zscore", False)):
        command.append("--per_run_zscore")
    _add_flag(command, "--run_tag", run_tag)
    _add_flag(command, "--output_base_dir", Path(output_base).resolve())
    return command, run_tag


def deduplicate_records(records: list[dict]) -> list[dict]:
    by_run: dict[tuple[str, str], dict] = {}
    for record in records:
        key = (str(record["subject"]), str(record.get("run_tag", record.get("status", "audit"))))
        existing = by_run.get(key)
        if existing is None:
            by_run[key] = record
            continue
        current_rank = STATUS_RANK.get(str(record.get("status")), -10)
        existing_rank = STATUS_RANK.get(str(existing.get("status")), -10)
        if current_rank >= existing_rank:
            by_run[key] = record
    return list(by_run.values())


def status_path(output_base: Path) -> Path:
    return Path(output_base) / STATUS_FILENAME


def write_status(output_base: Path, records: list[dict]) -> None:
    output_base = Path(output_base)
    output_base.mkdir(parents=True, exist_ok=True)
    records = deduplicate_records(records)
    payload = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "cohort": COHORT_LABEL,
        "interpretation": (
            "prespecified before model outcomes were computed; requires non-overlap with the current 33+20 subjects, "
            "360-column finite model inputs, and per-subject provenance before training. "
            "This is an HCP-derived evaluation, not an external cohort."
        ),
        "n_subjects": len(records),
        "n_eligible": sum(1 for record in records if record.get("eligible_for_training") is True),
        "n_missing_model_input": sum(1 for record in records if record.get("status") == "missing_model_input"),
        "n_missing_provenance": sum(1 for record in records if record.get("status") == "missing_provenance"),
        "n_invalid": sum(1 for record in records if str(record.get("status", "")).startswith("invalid_")),
        "n_success": sum(1 for record in records if record.get("status") == "success"),
        "n_complete": sum(1 for record in records if record.get("status") in COMPLETE_STATUSES),
        "records": records,
    }
    status_path(output_base).write_text(json.dumps(payload, indent=2), encoding="utf-8")


def required_outputs_exist(out_dir: Path) -> bool:
    return heldout_diag.required_outputs_exist(out_dir)


def enrich_record_from_outputs(record: dict, out_dir: Path) -> dict:
    return heldout_diag.enrich_record_from_outputs(record, out_dir)


def audit_all_inputs(
    subjects: list[str],
    data_dir: Path = DATA_DIR,
    provenance_dir: Path = DEFAULT_PROVENANCE_DIR,
) -> list[dict]:
    return [
        audit_subject_input(subject, data_dir=data_dir, provenance_dir=provenance_dir)
        for subject in subjects
    ]


def run_subject(
    subject: str,
    output_base: Path,
    data_dir: Path = DATA_DIR,
    provenance_dir: Path = DEFAULT_PROVENANCE_DIR,
    dry_run: bool = False,
) -> dict:
    audit = audit_subject_input(subject, data_dir=data_dir, provenance_dir=provenance_dir)
    if not audit.get("eligible_for_training"):
        return audit
    output_base = Path(output_base).resolve()
    config = build_prespecified_fixed_config(subject, data_path=data_path_for_subject(subject, data_dir))
    command, run_tag = build_training_command(subject, config, output_base)
    out_dir = output_base / subject / run_tag
    record = {
        **audit,
        "run_tag": run_tag,
        "status": "dry_run" if dry_run else "started",
        "command": command,
        "output_dir": str(out_dir.relative_to(ROOT)),
        "started_utc": datetime.now(timezone.utc).isoformat(),
    }
    if dry_run:
        return record

    started = time.time()
    completed = subprocess.run(command, cwd=str(ROOT), check=False)
    record["elapsed_seconds"] = float(time.time() - started)
    record["return_code"] = int(completed.returncode)
    if completed.returncode == 0:
        record["status"] = "success"
        record = enrich_record_from_outputs(record, out_dir)
    else:
        if required_outputs_exist(out_dir):
            record["status"] = "completed_with_nonzero_return"
            record["output_integrity_note"] = (
                "All required diagnostic outputs exist despite a non-zero process return code."
            )
            record = enrich_record_from_outputs(record, out_dir)
        else:
            record["status"] = "failed"
    record["finished_utc"] = datetime.now(timezone.utc).isoformat()
    return record


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subjects", nargs="+", default=PRESPECIFIED_SUBJECTS)
    parser.add_argument("--data_dir", type=Path, default=DATA_DIR)
    parser.add_argument("--provenance_dir", type=Path, default=DEFAULT_PROVENANCE_DIR)
    parser.add_argument("--output_base", type=Path, default=DEFAULT_OUTPUT_BASE)
    parser.add_argument("--audit_only", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    records = audit_all_inputs(
        [str(subject) for subject in args.subjects],
        data_dir=args.data_dir,
        provenance_dir=args.provenance_dir,
    )
    if args.audit_only:
        write_status(args.output_base, records)
        print(json.dumps({"status": "audit_only", "output": str(status_path(args.output_base))}, indent=2))
        return

    run_records = []
    for record in records:
        subject = str(record["subject"])
        if not record.get("eligible_for_training"):
            run_records.append(record)
            continue
        run_records.append(
            run_subject(
                subject,
                output_base=args.output_base,
                data_dir=args.data_dir,
                provenance_dir=args.provenance_dir,
                dry_run=bool(args.dry_run),
            )
        )
        write_status(args.output_base, run_records)
    write_status(args.output_base, run_records)
    print(json.dumps({"status": "finished", "output": str(status_path(args.output_base))}, indent=2))


if __name__ == "__main__":
    main()
