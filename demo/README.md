# Demo

This directory contains a lightweight simulated-data demo for reviewer
verification.

## Input

- `../demo_data/demo_simulated_360_timeseries.npy`

## Run

```bash
python run_demo.py --input ../demo_data/demo_simulated_360_timeseries.npy --output-dir ../demo_output
```

## Expected output

- `../demo_output/demo_summary.json`
- `../demo_output/demo_region_summary.csv`
- `../demo_output/demo_fc_preview.csv`
- `../demo_output/demo_fc_preview.png`

The checked-in reference outputs are stored in `expected_output/`.

The demo is simulated only. It is not used as scientific evidence for the
manuscript.
