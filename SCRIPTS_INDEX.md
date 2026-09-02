# Scripts Index

The scripts below are copied for audit and selected reruns. Some require the full local project environment and data access governed by HCP or upstream data-use terms.

## HCP-YA input preparation and evaluation

- `scripts/prepare_prespecified_hcp_inputs.py`: parcellates retained HCP-YA resting-state CIFTI files into 360-column model inputs and writes provenance JSON files.
- `scripts/run_prespecified_hcp_evaluation.py`: runs the fixed HCP-YA evaluation configuration on prepared 360-column inputs.

## Paired ablation

- `scripts/prepare_new25_paired_ablation_plan.py`: creates the fixed paired-ablation plan for the new 25-subject HCP-YA batch.
- `scripts/run_new25_paired_ablation_batch.py`: executes the paired-ablation runs.
- `scripts/summarize_new25_paired_ablation.py`: summarizes the paired-ablation outputs.
- `scripts/generate_new25_paired_ablation_figure.py`: generates the main paired-ablation figure.

## Main and supplementary figures

- `scripts/generate_reproduce0866_model_analysis_figure.py`: generates the held-out model-analysis figure.
- `scripts/generate_figure5_evidence_summary.py`: generates the evidence-level summary figure.
- `scripts/generate_supplementary_npi_workflow_figure.py`: generates the code-only schematic NPI workflow figure.

## Boundary

The copied scripts are not a claim that every path can run on a clean machine without HCP/F-TRACT access or the original project environment. They provide traceability from manuscript claims to local analysis code and summary outputs.
