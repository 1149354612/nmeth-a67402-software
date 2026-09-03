#!/usr/bin/env python
"""Run the new-25 paired ablation plan with checkpointed status files."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PLAN = ROOT / "new25_paired_ablation_plan" / "new25_paired_ablation_plan.json"
DEFAULT_STATUS = ROOT / "new25_paired_ablation_plan" / "batch_status.json"


def cell_outputs_exist(cell: dict) -> bool:
    outputs = cell.get("expected_key_outputs", [])
    return bool(outputs) and all(Path(path).exists() for path in outputs)


def write_status(status: dict, status_path: Path) -> None:
    status_path = Path(status_path)
    status_path.parent.mkdir(parents=True, exist_ok=True)
    status_path.write_text(json.dumps(status, indent=2), encoding="utf-8")


def run_cells(
    cells: list[dict],
    status_path: Path = DEFAULT_STATUS,
    timeout_seconds: float = 6 * 60 * 60,
    log_dir: Path | None = None,
) -> dict:
    status_path = Path(status_path)
    log_dir = Path(log_dir) if log_dir is not None else status_path.parent / "batch_logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    status = {
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "status_path": str(status_path),
        "n_requested": len(cells),
        "n_completed": 0,
        "n_failed": 0,
        "records": [],
        "interpretation_boundary": (
            "These are execution records only. Ablation effects are not manuscript results "
            "until all planned cells complete and paired statistics are computed."
        ),
    }
    write_status(status, status_path)

    for index, cell in enumerate(cells, start=1):
        subject = str(cell.get("subject", "unknown"))
        config = str(cell.get("config", "unknown"))
        stem = f"{index:03d}_{subject}_{config}"
        stdout_path = log_dir / f"{stem}.stdout.log"
        stderr_path = log_dir / f"{stem}.stderr.log"
        record = {
            "index": index,
            "subject": subject,
            "config": config,
            "started_utc": datetime.now(timezone.utc).isoformat(),
            "expected_key_outputs": cell.get("expected_key_outputs", []),
            "stdout_log": str(stdout_path),
            "stderr_log": str(stderr_path),
            "skipped_existing_outputs": False,
            "returncode": None,
            "elapsed_seconds": None,
        }

        if cell_outputs_exist(cell):
            record["skipped_existing_outputs"] = True
            record["returncode"] = 0
            record["elapsed_seconds"] = 0.0
            record["finished_utc"] = datetime.now(timezone.utc).isoformat()
            status["records"].append(record)
            status["n_completed"] += 1
            write_status(status, status_path)
            continue

        command = cell.get("command", [])
        if not command:
            record["returncode"] = "missing_command"
            record["finished_utc"] = datetime.now(timezone.utc).isoformat()
            status["records"].append(record)
            status["n_failed"] += 1
            status["stopped_after_failure"] = {"subject": subject, "config": config}
            write_status(status, status_path)
            return status

        start = time.time()
        with stdout_path.open("w", encoding="utf-8") as stdout_f, stderr_path.open("w", encoding="utf-8") as stderr_f:
            try:
                completed = subprocess.run(
                    command,
                    cwd=str(ROOT),
                    stdout=stdout_f,
                    stderr=stderr_f,
                    timeout=timeout_seconds,
                )
                record["returncode"] = completed.returncode
            except subprocess.TimeoutExpired:
                record["returncode"] = "timeout"

        record["elapsed_seconds"] = round(time.time() - start, 2)
        record["finished_utc"] = datetime.now(timezone.utc).isoformat()
        record["key_outputs_exist"] = cell_outputs_exist(cell)
        status["records"].append(record)

        if record["returncode"] == 0 and record["key_outputs_exist"]:
            status["n_completed"] += 1
            write_status(status, status_path)
            continue

        status["n_failed"] += 1
        status["stopped_after_failure"] = {"subject": subject, "config": config}
        write_status(status, status_path)
        return status

    status["finished_utc"] = datetime.now(timezone.utc).isoformat()
    status["completed_all_requested_cells"] = True
    write_status(status, status_path)
    return status


def load_train_cells(plan_path: Path) -> list[dict]:
    plan = json.loads(Path(plan_path).read_text(encoding="utf-8"))
    return [cell for cell in plan.get("cells", []) if cell.get("cell_type") == "train_ablation"]


def filter_cells(
    cells: list[dict],
    subjects: list[str] | None = None,
    configs: list[str] | None = None,
    max_cells: int | None = None,
) -> list[dict]:
    selected = cells
    if subjects:
        subject_set = {str(subject) for subject in subjects}
        selected = [cell for cell in selected if str(cell.get("subject")) in subject_set]
    if configs:
        config_set = {str(config) for config in configs}
        selected = [cell for cell in selected if str(cell.get("config")) in config_set]
    if max_cells is not None:
        selected = selected[:max_cells]
    return selected


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run selected cells from the new-25 paired ablation plan.")
    parser.add_argument("--plan", type=Path, default=DEFAULT_PLAN)
    parser.add_argument("--status", type=Path, default=DEFAULT_STATUS)
    parser.add_argument("--subjects", nargs="*", default=None)
    parser.add_argument("--configs", nargs="*", default=None)
    parser.add_argument("--max-cells", type=int, default=None)
    parser.add_argument("--timeout-minutes", type=float, default=360.0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    cells = filter_cells(
        load_train_cells(args.plan),
        subjects=args.subjects,
        configs=args.configs,
        max_cells=args.max_cells,
    )
    status = run_cells(
        cells,
        status_path=args.status,
        timeout_seconds=args.timeout_minutes * 60,
    )
    print(json.dumps(status, indent=2))
    return 0 if status.get("n_failed", 0) == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
