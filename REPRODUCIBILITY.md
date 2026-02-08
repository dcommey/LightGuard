# Reproducibility

This document explains how to obtain datasets and reproduce the main results.

## Environment

Windows PowerShell:

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Datasets

This repo’s `.gitignore` excludes `data/` and `results/` by default.

### CIC-Darknet-2020

Preset key: `darknet`.

Option A (recommended): place the CSV at:

- `data/Darknet.csv`

Option B: download via the runner:

```powershell
.\venv\Scripts\python.exe run_analysis.py --dataset darknet --prepare-only
```

To re-download:

```powershell
.\venv\Scripts\python.exe run_analysis.py --dataset darknet --prepare-only --force-download
```

### ISCX-VPN-NonVPN-2016

Preset key: `iscx_vpn_nonvpn_2016`.

The code downloads scenario `*-ARFF.zip` archives and converts ARFF → CSV.

```powershell
.\venv\Scripts\python.exe run_analysis.py --dataset iscx_vpn_nonvpn_2016 --prepare-only
```

### CSE-CIC-IDS2018 (AWS S3 mirror)

Preset key: `cse_cic_ids2018`.

This dataset is distributed as a public AWS S3 bucket (large; no single CSV URL). Install the AWS CLI and run:

```powershell
aws s3 sync --no-sign-request --region ca-central-1 "s3://cse-cic-ids2018/" data\cse_cic_ids2018
```

## Quick validation (no experiments)

This checks that the dataset loads and preprocesses correctly.

```powershell
.\venv\Scripts\python.exe run_analysis.py --dataset darknet --validate-only
```

## Running experiments

### Single run

```powershell
.\venv\Scripts\python.exe run_analysis.py --dataset darknet --output-dir results/darknet_run
```

### Multi-seed mean ± 95% CI

```powershell
.\venv\Scripts\python.exe run_analysis.py --dataset darknet --seeds 1,2,3 --output-dir results/darknet_ci
```

### Aggregate-only (rebuild CI outputs from existing seeds)

```powershell
.\venv\Scripts\python.exe run_analysis.py --output-dir results/darknet_ci --aggregate-only
```

## Robustness + diagnostics

You can run these either directly via `scripts/` or through the `run_analysis.py diag ...` subcommands.

### MMD probe-size robustness (100 vs 1k vs 5k)

This runs a small Monte Carlo check of the MMD drift score stability as the probe size increases.
It writes per-window means/stds plus drift-decision agreement vs the default 100-flow probe.

```powershell
.\venv\Scripts\python.exe scripts\probe_size_robustness.py --dataset darknet --probe-sizes 100,1000,5000 --reps 50 --seed 1 --output-dir results\darknet_ci\seed_1\diagnostics
```

Equivalent via runner subcommand:

```powershell
.\venv\Scripts\python.exe run_analysis.py diag probe-robustness --dataset darknet --probe-sizes 100,1000,5000 --reps 50 --seed 1 --output-dir results\darknet_ci\seed_1\diagnostics
```

Outputs:

- `results/darknet_ci/seed_1/diagnostics/mmd_probe_robustness.csv`
- `results/darknet_ci/seed_1/diagnostics/mmd_probe_robustness_summary.csv`

### Accuracy diagnostics (more decimals + confusion matrices)

This exports window-level accuracies to 6 decimals and writes confusion matrices for LightGuard windows whose accuracy rounds to `1.000` at 3 decimals.

```powershell
.\venv\Scripts\python.exe scripts\accuracy_diagnostics.py --dataset darknet --seed 1 --output-dir results\darknet_ci\seed_1\diagnostics
```

Equivalent via runner subcommand:

```powershell
.\venv\Scripts\python.exe run_analysis.py diag accuracy --dataset darknet --seed 1 --output-dir results\darknet_ci\seed_1\diagnostics
```

Outputs:

- `results/darknet_ci/seed_1/diagnostics/accuracy_diagnostics.csv`
- `results/darknet_ci/seed_1/diagnostics/accuracy_diagnostics_lightguard_summary.csv`
- `results/darknet_ci/seed_1/diagnostics/lightguard_confusion_matrix_window_<k>.csv`

## Notes

- Chronological windows are strictly non-overlapping, and evaluation is always on future windows relative to the model state being evaluated.
- Some printed metric values are rounded (e.g., `1.000` in tables means 0.9995+ at 3 decimals, not necessarily perfect to all digits).
