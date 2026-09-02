# Nature Methods Software Package

This folder is the reviewer-facing software package for NMETH-A67402,
`Virtual Neural Perturbation with Anatomically Informed Transformer Surrogates
for Subject-Specific Cortical Effective Connectivity Estimation from
Resting-State fMRI`.

## Included

- `scripts/`: selected analysis and figure-generation scripts.
- `summary_outputs/`: non-identifying summary outputs used in the manuscript.
- `provenance_json/`: per-subject provenance records for the traceable HCP-YA batch.
- `subject_lists/`: fixed subject lists used by the reported analyses.
- `demo_data/`: simulated data used only to verify that the software runs.
- `demo/`: a lightweight demo runner and the expected demo outputs.
- `RUNBOOK_REPRODUCE_RESULTS.md`: reviewer audit path for reported values.
- `DATA_USE_AND_RESTRICTIONS.md`: data-use boundaries.
- `REPRODUCIBILITY_PACKAGE_MANIFEST.md`: package-level evidence accounting.

## What this package is for

This package supports reviewer inspection of the code and the reported summary
outputs. It is not a redistribution of restricted HCP or F-TRACT source data.
The demo data are simulated and exist only to verify execution.

## System requirements

- Operating system: Windows 11 or Ubuntu 22.04
- Demo verified on Windows 11 with Python 3.13.5
- Full analysis environment specification is provided for Python 3.11
- Core demo dependencies: `numpy`, `pandas`, `scipy`, `matplotlib`, `seaborn`
- Full analysis dependencies: PyTorch 2.0 series, `scikit-learn`, `nibabel`, `nilearn`,
  `tqdm`, `h5py`, `pyyaml`, `joblib`, `click`, `rich`, `loguru`, `omegaconf`
- Connectome Workbench 2.1.0 is required for the HCP input-preparation scripts
- No non-standard hardware is required for the simulated demo
- GPU hardware is recommended for the full model scripts

## Installation

From this folder:

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Typical install time on a normal desktop computer: about 10-30 minutes,
depending on network speed and what is already installed locally.

## Demo

Run the simulated demo with:

```bash
python demo/run_demo.py --input demo_data/demo_simulated_360_timeseries.npy --output-dir demo_output
```

Expected output files:

- `demo_output/demo_summary.json`
- `demo_output/demo_region_summary.csv`
- `demo_output/demo_fc_preview.csv`
- `demo_output/demo_fc_preview.png`

Checked-in reference outputs are stored in `demo/expected_output/`.

Expected demo runtime on a normal desktop computer: about 1-2 minutes.

The demo dataset is simulated and is not used as scientific evidence for the
manuscript.

## Running on manuscript data

The selected manuscript scripts expect HCP-derived 360-column arrays and the
matching per-subject provenance JSON files:

- `scripts/prepare_prespecified_hcp_inputs.py`
- `scripts/run_prespecified_hcp_evaluation.py`
- `scripts/prepare_new25_paired_ablation_plan.py`
- `scripts/run_new25_paired_ablation_batch.py`
- `scripts/summarize_new25_paired_ablation.py`

For the current reviewer package, use `RUNBOOK_REPRODUCE_RESULTS.md` to inspect
the reported values and evidence boundaries.

## Data restrictions

Restricted HCP or F-TRACT source imaging data are not redistributed here.
Source-derived data access remains governed by the original data-use terms.
