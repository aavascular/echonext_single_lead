# EchoNext Single-Lead Benchmark

This repository provides a reproducible PyTorch benchmark for evaluating whether reduced-lead ECG inputs, especially single-lead ECGs, can detect echocardiogram-confirmed structural heart disease (SHD) using the EchoNext Mini-Model dataset.

## Project goal

The benchmark compares:

- Full 12-lead ECG models
- Lead I-only ECG models as a wearable-compatible proxy
- Each individual single-lead ECG model
- Lead I plus tabular feature models
- Optional label-efficiency experiments using smaller training fractions

The first proof-of-concept focuses on:

- `shd_moderate_or_greater_flag`
- `lvef_lte_45_flag`

The pipeline is written to generalize to the broader label set listed in [`configs/config.yaml`](/Users/aaljobeh/Documents/EchoNext/echonext_single_lead/configs/config.yaml).

## Expected dataset files

Place the following files inside `data/raw/`:

- `echonext_metadata_100k.csv`
- `EchoNext_train_waveforms.npy`
- `EchoNext_val_waveforms.npy`
- `EchoNext_test_waveforms.npy`
- `EchoNext_no_split_waveforms.npy`
- `EchoNext_train_tabular_features.npy`
- `EchoNext_val_tabular_features.npy`
- `EchoNext_test_tabular_features.npy`
- `EchoNext_no_split_tabular_features.npy`

The first version uses only the `train`, `val`, and `test` splits. The `no_split` files are ignored.

## Directory layout

```text
echonext_single_lead/
├── README.md
├── requirements.txt
├── configs/
├── data/
│   ├── raw/
│   └── processed/
├── outputs/
│   ├── figures/
│   ├── models/
│   ├── predictions/
│   └── tables/
├── scripts/
└── src/
```

## Installation

Python 3.10+ is required.

```bash
cd echonext_single_lead
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

For Google Colab, upload or mount the project and install requirements with:

```bash
pip install -r requirements.txt
```

## How to run

Colab starter notebook:

[`notebooks/01_colab_starter.ipynb`](/Users/aaljobeh/Documents/EchoNext/echonext_single_lead/notebooks/01_colab_starter.ipynb)

Inspect the dataset:

```bash
python scripts/01_inspect_data.py --data_dir data/raw
```

Train a tabular baseline:

```bash
python scripts/02_train_baseline_tabular.py --data_dir data/raw --label shd_moderate_or_greater_flag
```

Train a Lead I model:

```bash
python scripts/03_train_ecg_model.py --data_dir data/raw --label shd_moderate_or_greater_flag --input_mode single --lead I
```

Train a full 12-lead model:

```bash
python scripts/03_train_ecg_model.py --data_dir data/raw --label shd_moderate_or_greater_flag --input_mode full12
```

Run the lead sweep benchmark:

```bash
python scripts/04_run_lead_sweep.py --data_dir data/raw
```

Run the label-efficiency experiment:

```bash
python scripts/05_run_label_efficiency.py --data_dir data/raw --label shd_moderate_or_greater_flag
```

Build summary tables and figures from completed runs:

```bash
python scripts/06_make_results_tables.py --output_dir outputs
```

## Output files

Training runs create a dedicated run directory under `outputs/models/` and companion prediction files under `outputs/predictions/`.

Each run saves:

- `best_model.pt`: best checkpoint selected on validation performance
- `training_log.csv`: epoch-level losses and validation metrics
- `val_predictions.csv`: validation predictions
- `test_predictions.csv`: test predictions
- `metrics.json`: validation and test metrics plus run metadata

The aggregation script creates:

- `outputs/tables/model_performance_by_input.csv`
- `outputs/tables/label_efficiency_results.csv`
- `outputs/figures/auroc_by_input.png`
- `outputs/figures/auprc_by_input.png`
- `outputs/figures/label_efficiency_auroc.png`
- `outputs/figures/calibration_plot.png`

## Notes

- The metadata `split` column is respected.
- Row order is preserved within each split so metadata aligns with waveform and tabular arrays.
- Waveforms are loaded with NumPy memory mapping to avoid loading the full array into RAM.
- Validation and test sets remain fixed across experiments.
- The training script reports prevalence for each split before fitting.
- Class imbalance is handled via `pos_weight` in the binary loss.

If your metadata uses different identifier columns, update the config file rather than editing the code directly.
