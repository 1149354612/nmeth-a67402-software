# Data Use and Redistribution Boundaries

This project uses de-identified, previously released HCP-derived resting-state fMRI resources and aggregate F-TRACT CCEP resources. No new human participants were recruited, and no new stimulation or imaging data were collected for this study.

## HCP-derived data

The HCP source imaging data and derived files remain subject to HCP data-use terms. The local workspace contains processed 360-column arrays and provenance files for the fixed HCP-YA robustness evaluation. The reviewer package does not redistribute source CIFTI/NIfTI files or processed `.npy` model-input arrays by default.

Reviewer-facing materials include subject identifiers, provenance JSON files, checksums, commands, and summary outputs. Reuse or redistribution of source-derived data should follow the applicable HCP access terms.

## F-TRACT CCEP data

CCEP analyses use aggregate F-TRACT resources available in the project. The manuscript does not use same-subject resting-fMRI plus invasive stimulation-response recordings. CCEP correspondence is reported as a sensitivity analysis and limitation rather than biological ground truth.

## Upstream NPI resources

The NPI concept and reference public repository/resource were released by Luo et al. The manuscript distinguishes public NPI-derived resources from locally processed HCP-derived files. Public release or redistribution of upstream source-derived files remains subject to upstream access terms.

## Package boundary

This reviewer package includes provenance, commands, selected scripts, summary
outputs, figure metadata, and a small simulated demo dataset used only for
execution checking. It excludes restricted source imaging, model checkpoints,
and any redistributed HCP or F-TRACT source arrays. This avoids overstating
data-sharing rights while preserving an auditable trail for the manuscript
values.
