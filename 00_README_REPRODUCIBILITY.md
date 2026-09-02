# Reviewer Reproducibility Package

Prepared: 2026-07-16

This package supports reviewer inspection of the manuscript:

`Virtual Neural Perturbation with Anatomically Informed Transformer Surrogates for Subject-Specific Cortical Effective Connectivity Estimation from Resting-State fMRI`

The package is designed for auditability. It contains the manuscript provenance index, subject lists, non-identifying summary outputs, provenance JSON files, figure metadata, and scripts needed to inspect or rerun selected analyses in the local project environment.

## What is included

- `REPRODUCIBILITY_PACKAGE_MANIFEST.md`: current file-level manifest and evidence boundaries.
- `SUPPLEMENTARY_TABLE_S1_ANALYSIS_PROVENANCE.md`: provenance and evidence-level accounting table used by the supplementary information.
- `RUNBOOK_REPRODUCE_RESULTS.md`: shortest path for checking the reported values.
- `DATA_USE_AND_RESTRICTIONS.md`: data-use boundaries for HCP, F-TRACT, and processed outputs.
- `subject_lists/`: fixed 25-subject list, 27 processed-subject list, and two processed-but-not-statistical subjects.
- `provenance_json/`: per-subject provenance JSON files for the 27 processed HCP-YA subjects.
- `summary_outputs/`: selected non-identifying summary outputs used by the manuscript.
- `figure_metadata/`: metadata for code-generated or data-summary figures.
- `scripts/`: selected scripts used for input preparation, evaluation, ablation, and figure generation.
- `demo_data/`: a small simulated dataset used only to verify software execution.
- `demo/`: a lightweight demo runner and the expected demo outputs.

## What is not included

No restricted HCP or F-TRACT source imaging data are redistributed here. No
source CIFTI, NIfTI, checkpoint, or PyTorch model-weight files are included.
The only `.npy` file added to the package is a small simulated demo input that
is used solely to verify execution. Access to source-derived data remains
governed by the original HCP, F-TRACT, and upstream repository terms.

## Main reproducibility boundary

The reviewer package supports verification of provenance, commands, subject inclusion, summary outputs, and analysis logic. It is not a complete public redistribution of all source data needed to reproduce every training run from scratch without HCP/F-TRACT access.
