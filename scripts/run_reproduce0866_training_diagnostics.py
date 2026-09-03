"""Rerun fixed held-out reproduce_0866 training with epoch-level diagnostics.

The original fixed held-out directories contain final checkpoints and output
matrices but no per-epoch training history. This script reruns the same fixed
configuration into a separate diagnostics directory so training curves can be
reported as a rerun diagnostic without overwriting the manuscript results.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
VALIDATION_DIR = ROOT / "validation_results"
DEFAULT_OUTPUT_BASE = ROOT / "training_diagnostics_rerun"
SUBJECTS = ["106824", "117021", "123723", "136631", "151930", "152225"]
REQUIRED_OUTPUTS = [
    "best_model.pth",
    "training_history.json",
    "training_history.csv",
    "training_summary.json",
    "empirical_fc.npy",
    "model_fc.npy",
    "fc_r.txt",
    "ec.npy",
    "ec_stats.json",
    "config.json",
]
COMPLETE_STATUSES = {"success", "completed_with_nonzero_return"}
STATUS_RANK = {
    "dry_run": 0,
    "started": 1,
    "failed": 2,
    "completed_with_nonzero_return": 3,
    "success": 4,
}


def load_fixed_config(subject: str) -> dict:
    path = VALIDATION_DIR / subject / f"reproduce_0866_{subject}" / "config.json"
    if not path.exists():
        raise FileNotFoundError(f"Missing fixed reproduce_0866 config: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _add_flag(command: list[str], name: str, value) -> None:
    command.extend([name, str(value)])


def build_training_command(subject: str, config: dict, output_base: Path) -> tuple[list[str], str]:
    data_path = Path(config["data_path"])
    if not data_path.exists():
        raise FileNotFoundError(f"Missing data for subject {subject}: {data_path}")
    run_tag = f"training_diagnostics_reproduce_0866_{subject}"
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
    _add_flag(command, "--output_base_dir", output_base.resolve())
    return command, run_tag


def deduplicate_records(records: list[dict]) -> list[dict]:
    by_run: dict[tuple[str, str], dict] = {}
    for record in records:
        key = (str(record["subject"]), str(record["run_tag"]))
        existing = by_run.get(key)
        if existing is None:
            by_run[key] = record
            continue
        current_rank = STATUS_RANK.get(str(record.get("status")), -1)
        existing_rank = STATUS_RANK.get(str(existing.get("status")), -1)
        if current_rank >= existing_rank:
            by_run[key] = record
    return list(by_run.values())


def write_status(output_base: Path, records: list[dict]) -> None:
    output_base.mkdir(parents=True, exist_ok=True)
    records = deduplicate_records(records)
    payload = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "n_subjects": len(records),
        "n_success": sum(1 for record in records if record.get("status") == "success"),
        "n_completed_with_nonzero_return": sum(
            1 for record in records if record.get("status") == "completed_with_nonzero_return"
        ),
        "n_complete": sum(1 for record in records if record.get("status") in COMPLETE_STATUSES),
        "n_failed": sum(1 for record in records if record.get("status") == "failed"),
        "records": records,
    }
    (output_base / "training_diagnostics_status.json").write_text(
        json.dumps(payload, indent=2),
        encoding="utf-8",
    )


def required_outputs_exist(out_dir: Path) -> bool:
    return all((out_dir / name).exists() for name in REQUIRED_OUTPUTS)


def enrich_record_from_outputs(record: dict, out_dir: Path) -> dict:
    fc_path = out_dir / "fc_r.txt"
    if fc_path.exists():
        record["fc_r"] = float(fc_path.read_text(encoding="utf-8").strip())
    summary_path = out_dir / "training_summary.json"
    if summary_path.exists():
        record["training_summary"] = json.loads(summary_path.read_text(encoding="utf-8"))
    return record


def read_existing_status(output_base: Path) -> list[dict]:
    status_path = output_base / "training_diagnostics_status.json"
    if not status_path.exists():
        return []
    payload = json.loads(status_path.read_text(encoding="utf-8"))
    return list(payload.get("records", []))


def reconcile_existing_records(records: list[dict], output_base: Path) -> list[dict]:
    reconciled = []
    for record in records:
        subject = str(record["subject"])
        run_tag = str(record["run_tag"])
        out_dir = output_base / subject / run_tag
        if record.get("status") != "success" and required_outputs_exist(out_dir):
            record = dict(record)
            record["status"] = "completed_with_nonzero_return"
            record["output_integrity_note"] = (
                "All required diagnostic outputs exist despite a non-zero process return code."
            )
        if required_outputs_exist(out_dir):
            record = enrich_record_from_outputs(record, out_dir)
        reconciled.append(record)
    return deduplicate_records(reconciled)


def run_subject(subject: str, output_base: Path, dry_run: bool = False) -> dict:
    output_base = Path(output_base).resolve()
    config = load_fixed_config(subject)
    command, run_tag = build_training_command(subject, config, output_base)
    out_dir = output_base / subject / run_tag
    record = {
        "subject": subject,
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
    parser.add_argument("--subjects", nargs="+", default=SUBJECTS)
    parser.add_argument("--output_base", type=Path, default=DEFAULT_OUTPUT_BASE)
    parser.add_argument("--dry_run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_base = Path(args.output_base).resolve()
    records = reconcile_existing_records(read_existing_status(output_base), output_base)
    completed_subjects = {
        str(record["subject"])
        for record in records
        if record.get("status") in COMPLETE_STATUSES
    }
    if records:
        write_status(output_base, records)
    for subject in args.subjects:
        if str(subject) in completed_subjects:
            print(f"[training diagnostics] subject {subject} already has complete outputs; skipping")
            continue
        print(f"[training diagnostics] subject {subject}")
        record = run_subject(str(subject), output_base=output_base, dry_run=bool(args.dry_run))
        records.append(record)
        write_status(output_base, records)
        print(json.dumps({k: v for k, v in record.items() if k != "command"}, indent=2))
        if record["status"] == "failed":
            break


if __name__ == "__main__":
    main()
