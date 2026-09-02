#!/usr/bin/env python
"""Summarize completed new-25 paired ablation results.

This script reads the frozen paired-ablation plan, requires every subject to
have the full reference condition and every ablation condition, and writes
per-subject and paired-statistics outputs. It does not infer missing values.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from statistics import mean, stdev


ROOT = Path(__file__).resolve().parent
DEFAULT_PLAN = ROOT / "new25_paired_ablation_plan" / "new25_paired_ablation_plan.json"
DEFAULT_OUTPUT_DIR = ROOT / "new25_paired_ablation_plan" / "summary"
REFERENCE_CONFIG = "full_mgka_h60_fc05"


def load_plan(path: Path = DEFAULT_PLAN) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def resolve_path(path_text: str, root: Path = ROOT) -> Path:
    path = Path(path_text)
    return path if path.is_absolute() else Path(root) / path


def fc_path_for_cell(cell: dict, root: Path = ROOT) -> Path:
    for output in cell.get("expected_key_outputs", []):
        if str(output).replace("\\", "/").endswith("fc_r.txt"):
            return resolve_path(str(output), root=root)
    if cell.get("expected_output_dir"):
        return resolve_path(str(cell["expected_output_dir"]), root=root) / "fc_r.txt"
    raise ValueError(f"missing fc_r path for subject={cell.get('subject')} config={cell.get('config')}")


def read_fc_value(path: Path) -> float:
    text = Path(path).read_text(encoding="utf-8").strip()
    value = float(text)
    if not math.isfinite(value):
        raise ValueError(f"non-finite fc_r in {path}")
    return value


def condition_names(plan: dict) -> list[str]:
    names = [str(config["name"]) for config in plan.get("configs", []) if config.get("name")]
    if REFERENCE_CONFIG not in names:
        names.insert(0, REFERENCE_CONFIG)
    return names


def subject_names(plan: dict) -> list[str]:
    subjects = [str(subject) for subject in plan.get("subjects", [])]
    if not subjects:
        subjects = sorted({str(cell["subject"]) for cell in plan.get("cells", []) if cell.get("subject")})
    return subjects


def collect_fc_matrix(plan: dict, root: Path = ROOT) -> list[dict]:
    conditions = condition_names(plan)
    subjects = subject_names(plan)
    cell_map: dict[tuple[str, str], dict] = {}
    for cell in plan.get("cells", []):
        subject = str(cell.get("subject"))
        config = str(cell.get("config"))
        if subject and config:
            cell_map[(subject, config)] = cell

    rows = []
    for subject in subjects:
        row = {"subject": subject}
        for config in conditions:
            cell = cell_map.get((subject, config))
            if cell is None:
                raise ValueError(f"missing paired cell for subject={subject} config={config}")
            path = fc_path_for_cell(cell, root=root)
            if not path.exists():
                raise ValueError(f"missing fc_r file for subject={subject} config={config}: {path}")
            row[config] = read_fc_value(path)
        rows.append(row)
    return rows


def summarize_values(values: list[float]) -> dict:
    n = len(values)
    if n == 0:
        raise ValueError("cannot summarize empty values")
    sd = stdev(values) if n > 1 else 0.0
    return {
        "n": n,
        "mean": float(mean(values)),
        "sd": float(sd),
        "sem": float(sd / math.sqrt(n)) if n > 0 else None,
        "min": float(min(values)),
        "max": float(max(values)),
    }


def _bonferroni(p_value: float | None, family_size: int) -> float | None:
    if p_value is None:
        return None
    return min(1.0, float(p_value) * int(family_size))


def paired_test(reference: list[float], condition: list[float], family_size: int = 1) -> dict:
    if len(reference) != len(condition):
        raise ValueError("paired vectors must have the same length")
    n = len(reference)
    deltas_condition_minus_full = [condition[i] - reference[i] for i in range(n)]
    deltas_full_minus_condition = [-value for value in deltas_condition_minus_full]
    delta_summary = summarize_values(deltas_condition_minus_full)

    t_stat = None
    t_p = None
    wilcoxon_stat = None
    wilcoxon_p = None
    try:
        from scipy import stats

        t_res = stats.ttest_rel(condition, reference)
        t_stat = None if not math.isfinite(float(t_res.statistic)) else float(t_res.statistic)
        t_p = None if not math.isfinite(float(t_res.pvalue)) else float(t_res.pvalue)
        try:
            w_res = stats.wilcoxon(condition, reference, zero_method="wilcox", alternative="two-sided")
            wilcoxon_stat = float(w_res.statistic)
            wilcoxon_p = float(w_res.pvalue)
        except ValueError:
            wilcoxon_stat = None
            wilcoxon_p = None
    except Exception:
        pass

    return {
        "n": n,
        "mean_delta_condition_minus_full": float(mean(deltas_condition_minus_full)),
        "sd_delta_condition_minus_full": float(stdev(deltas_condition_minus_full)) if n > 1 else 0.0,
        "mean_delta_full_minus_condition": float(mean(deltas_full_minus_condition)),
        "sd_delta_full_minus_condition": float(stdev(deltas_full_minus_condition)) if n > 1 else 0.0,
        "delta_condition_minus_full_summary": delta_summary,
        "paired_t_statistic_condition_minus_full": t_stat,
        "paired_t_pvalue": t_p,
        "paired_t_pvalue_bonferroni": _bonferroni(t_p, family_size),
        "wilcoxon_statistic": wilcoxon_stat,
        "wilcoxon_pvalue": wilcoxon_p,
        "wilcoxon_pvalue_bonferroni": _bonferroni(wilcoxon_p, family_size),
        "bonferroni_family_size": int(family_size),
    }


def build_summary(plan: dict, root: Path = ROOT) -> dict:
    rows = collect_fc_matrix(plan, root=root)
    conditions = condition_names(plan)
    reference = [float(row[REFERENCE_CONFIG]) for row in rows]

    condition_summary = {
        config: summarize_values([float(row[config]) for row in rows])
        for config in conditions
    }
    paired_comparisons = {}
    ablation_conditions = [config for config in conditions if config != REFERENCE_CONFIG]
    family_size = len(ablation_conditions)
    for config in conditions:
        if config == REFERENCE_CONFIG:
            continue
        paired_comparisons[f"{config}_vs_{REFERENCE_CONFIG}"] = paired_test(
            reference,
            [float(row[config]) for row in rows],
            family_size=family_size,
        )

    return {
        "summary_type": "new25_paired_ablation_summary",
        "complete": True,
        "interpretation": (
            "Strictly paired ablation on the fixed-list traceable 25-subject HCP-YA batch. "
            "All conditions are HCP-derived post-configuration robustness analyses, not an external validation cohort."
        ),
        "n_subjects": len(rows),
        "reference_config": REFERENCE_CONFIG,
        "conditions": conditions,
        "per_subject": rows,
        "condition_summary": condition_summary,
        "paired_comparisons": paired_comparisons,
    }


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_outputs(result: dict, output_dir: Path = DEFAULT_OUTPUT_DIR) -> dict[str, Path]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_json = output_dir / "new25_paired_ablation_summary.json"
    per_subject_csv = output_dir / "new25_paired_ablation_per_subject.csv"
    condition_summary_csv = output_dir / "new25_paired_ablation_condition_summary.csv"
    paired_comparisons_csv = output_dir / "new25_paired_ablation_paired_comparisons.csv"

    summary_json.write_text(json.dumps(result, indent=2), encoding="utf-8")

    conditions = list(result["conditions"])
    write_csv(per_subject_csv, result["per_subject"], ["subject", *conditions])

    condition_rows = [
        {"condition": condition, **stats}
        for condition, stats in result["condition_summary"].items()
    ]
    write_csv(condition_summary_csv, condition_rows, ["condition", "n", "mean", "sd", "sem", "min", "max"])

    comparison_fieldnames = [
        "comparison",
        "n",
        "bonferroni_family_size",
        "mean_delta_condition_minus_full",
        "sd_delta_condition_minus_full",
        "mean_delta_full_minus_condition",
        "sd_delta_full_minus_condition",
        "paired_t_statistic_condition_minus_full",
        "paired_t_pvalue",
        "paired_t_pvalue_bonferroni",
        "wilcoxon_statistic",
        "wilcoxon_pvalue",
        "wilcoxon_pvalue_bonferroni",
    ]
    comparison_rows = [
        {field: (comparison if field == "comparison" else stats.get(field)) for field in comparison_fieldnames}
        for comparison, stats in result["paired_comparisons"].items()
    ]
    write_csv(
        paired_comparisons_csv,
        comparison_rows,
        comparison_fieldnames,
    )

    return {
        "summary_json": summary_json,
        "per_subject_csv": per_subject_csv,
        "condition_summary_csv": condition_summary_csv,
        "paired_comparisons_csv": paired_comparisons_csv,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize completed new-25 paired ablation outputs.")
    parser.add_argument("--plan", type=Path, default=DEFAULT_PLAN)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    plan = load_plan(args.plan)
    result = build_summary(plan, root=ROOT)
    paths = write_outputs(result, output_dir=args.output_dir)
    print(
        json.dumps(
            {
                "summary_json": str(paths["summary_json"]),
                "per_subject_csv": str(paths["per_subject_csv"]),
                "condition_summary_csv": str(paths["condition_summary_csv"]),
                "paired_comparisons_csv": str(paths["paired_comparisons_csv"]),
                "n_subjects": result["n_subjects"],
                "conditions": result["conditions"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
