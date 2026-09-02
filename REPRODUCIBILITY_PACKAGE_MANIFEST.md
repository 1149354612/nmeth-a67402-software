# Nature Methods Reviewer Reproducibility Package Manifest

Prepared: 2026-09-02

This package supports reviewer inspection of the manuscript entitled:

`Virtual Neural Perturbation with Anatomically Informed Transformer Surrogates for Subject-Specific Cortical Effective Connectivity Estimation from Resting-State fMRI`

It is an audit and selected-rerun package. It is not a complete redistribution of the restricted HCP or F-TRACT source data required to retrain every model from scratch.

## Included materials

- `00_README_REPRODUCIBILITY.md`: package purpose and scope.
- `DATA_USE_AND_RESTRICTIONS.md`: data access and redistribution boundaries.
- `SUPPLEMENTARY_TABLE_S1_ANALYSIS_PROVENANCE.md`: provenance and evidence-level accounting.
- `RUNBOOK_REPRODUCE_RESULTS.md`: shortest reviewer audit path.
- `SCRIPTS_INDEX.md`: selected scripts included in the package.
- `subject_lists/`: fixed subject lists used for the reported analyses.
- `provenance_json/`: per-subject provenance records for the traceable 25-subject HCP-YA batch.
- `summary_outputs/`: non-identifying summary outputs and analysis status records.
- `figure_metadata/`: metadata for the manuscript figures.
- `scripts/`: selected input-preparation, evaluation, ablation, and figure-generation scripts.
- `demo_data/`: a small simulated demo dataset used only for execution checking.
- `demo/`: a lightweight demo runner and expected outputs.

## Evidence accounting

- Main manuscript cohort: 33 HCP-derived subjects, including 27 development subjects and 6 fixed held-out subjects.
- Traceable HCP-YA robustness batch: 25 non-overlapping subjects; 25/25 completed.
- Paired ablation: 25 subjects and 3 single-factor ablations per subject; 75/75 requested cells completed.
- Paired ridge AR(1) baseline: 25/25 subjects completed.
- CCEP comparison: aggregate F-TRACT HCP-MMP1 summaries compared with six held-out model outputs; exploratory sensitivity analysis only.
- The separate 20-subject legacy batch and the project-level 78-subject provenance count are not primary performance denominators.

## Scientific boundaries

- Model-implied perturbational effective connectivity is reported as a subject-specific hypothesis-generating output, not biological ground truth.
- The CCEP comparison is limited by aggregate data, incomplete edge coverage, and cohort mismatch; it is not same-subject invasive validation.
- Training diagnostics are reruns and are not recovered original complete training histories.
- The included scripts and summary files support audit and selected reruns but do not guarantee clean-machine full retraining.
- The simulated demo dataset is not part of the manuscript evidence base and does not represent HCP or F-TRACT source data.

## Data and environment boundaries

- No restricted HCP or F-TRACT source imaging data are redistributed.
- No CIFTI/NIfTI source files, model checkpoints, PyTorch weights, or source
  model-input arrays are included.
- The only included `.npy` file is a small simulated demo input for execution
  checking.
- The manuscript reports PyTorch 2.0 for model implementation.
- Provenance records identify Connectome Workbench commands; the local audited Workbench installation was version 2.1.0.
- Absolute paths appearing in provenance or command records refer to the original local environment and are not portable paths for reviewers.
