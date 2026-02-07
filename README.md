# LightGuard

LightGuard is a research codebase for time-aware (chronological) durability evaluation of encrypted traffic classifiers under concept drift, plus a lightweight rehearsal-based update strategy.

## Quickstart

### 1) Environment

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### 2) Data

By default, runs use `data/` for datasets and `results/` for outputs.

- CIC-Darknet-2020 preset expects `data/Darknet.csv`.
- ISCX-VPN-NonVPN-2016 can be downloaded/prepared automatically (ARFF zips → CSV).
- CSE-CIC-IDS2018 requires `aws s3 sync` (see `lightguard/datasets.py`).

Note: this repo’s `.gitignore` excludes `data/` and `results/` to keep GitHub pushes small.

### 3) Validate a dataset (load + preprocess only)

```powershell
.\venv\Scripts\python.exe run_analysis.py --dataset darknet --validate-only
```

### 4) Run experiments

Single run:

```powershell
.\venv\Scripts\python.exe run_analysis.py --dataset darknet --output-dir results/darknet_run
```

Multi-seed runs with aggregated mean ± 95% CI outputs:

```powershell
.\venv\Scripts\python.exe run_analysis.py --dataset darknet --seeds 1,2,3 --output-dir results/darknet_ci
```

Regenerate aggregated CI plots/tables from existing `seed_<seed>` folders:

```powershell
.\venv\Scripts\python.exe run_analysis.py --output-dir results/darknet_ci --aggregate-only
```

## License
MIT (see LICENSE).
