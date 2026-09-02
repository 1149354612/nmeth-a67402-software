# Runbook for Checking Reported Results

This runbook gives a package-relative audit path. It does not require reviewers to trust absolute paths from the original local machine.

## 1. Check the 25-subject HCP-YA robustness batch

Inspect:

- `subject_lists/new25_included_subjects.txt`
- `summary_outputs/new25_full_summary_20260703.json`
- `summary_outputs/new25_full_subject_metrics_20260703.csv`
- `provenance_json/*_provenance.json`

Expected boundary:

- 25/25 subjects completed.
- FC reconstruction mean is approximately `r = 0.8598`, SD approximately `0.0824`.
- Subjects `146735` and `146836` were processed but excluded from the fixed 25-subject statistics.

## 2. Check paired ablation and baseline accounting

Inspect:

- `summary_outputs/new25_paired_ablation_summary.json`
- `summary_outputs/new25_paired_ablation_condition_summary.csv`
- `summary_outputs/new25_paired_ablation_paired_comparisons.csv`
- `summary_outputs/new25_paired_simple_baseline_batch_status.json`

Expected boundary:

- 75/75 ablation cells completed.
- MGKA attention is the most consistently supported component.
- FC-loss weighting and horizon length show weaker or mixed component-level evidence.
- The paired ridge AR(1) baseline completed for all 25 subjects and is not external validation.

## 3. Check FC, Granger-style, and CCEP comparisons

Inspect:

- `summary_outputs/fc_pc_summary_statistics.json`
- `summary_outputs/granger_summary_statistics.json`
- `summary_outputs/ccep_fixed_summary_stats.json`
- `summary_outputs/ccep_null_summary_stats.json`
- `summary_outputs/ccep_coverage_summary.json`
- `summary_outputs/ccep_correspondence_null_methods.md`

Expected boundary:

- The fixed CCEP correspondence is low, approximately `r = 0.032 +/- 0.007`.
- Null controls are sensitivity diagnostics, not biological validation.
- CCEP data are aggregate F-TRACT summaries and are not same-subject invasive recordings.

## 4. Check figure provenance

Inspect the JSON files under `figure_metadata/`. Schematic figures are identified as schematic, and the NPI workflow figure does not display empirical EC results.

## 5. Selected reruns

The copied scripts under `scripts/` can be inspected and may support selected reruns if the reviewer has access to the required HCP-derived inputs, F-TRACT resources, Workbench, and compatible Python/PyTorch environments. The package intentionally excludes restricted source data, checkpoints, and high-dimensional arrays, so a clean-machine full retraining is outside this package's claim.
