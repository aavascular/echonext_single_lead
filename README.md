# EchoNext Single-Lead Benchmark

This repository contains a reproducible PyTorch benchmark for evaluating how much diagnostic signal for echocardiogram-confirmed structural heart disease (SHD) survives ECG lead reduction in the public EchoNext Mini dataset.

The repository is designed to support:
- primary benchmarking of full 12-lead ECG, reduced-lead ECG, and tabular-feature models
- repeated-seed model stability analysis
- publication-style summary tables and figures
- Google Colab workflows that use GitHub for code and Google Drive for data and saved outputs

## Study Aim

The central study question is whether reduced-lead ECG configurations, especially wearable-compatible single-lead inputs such as Lead I, retain useful predictive signal for detecting echocardiogram-confirmed structural heart disease.

The benchmark currently supports comparison of:
- full 12-lead ECG
- six limb leads (`I`, `II`, `III`, `aVR`, `aVL`, `aVF`)
- each individual single lead
- tabular-only features
- Lead I plus tabular features

The primary outcome is:
- `shd_moderate_or_greater_flag`

Planned secondary outcomes include:
- `lvef_lte_45_flag`
- `rv_systolic_dysfunction_moderate_or_greater_flag`
- `lvwt_gte_13_flag`

## Repository Scope

This repository contains code, notebooks, configuration files, and documentation only. It does not redistribute the EchoNext dataset.

The expected workflow is:
- GitHub = source of truth for code
- Google Drive or local storage = source of truth for data and outputs
- Colab = temporary execution environment

## Data Use and Access

This repository does not include the EchoNext dataset or any derived patient-level prediction files. Users must obtain access to the dataset directly from the official EchoNext/PhysioNet source and are responsible for complying with the applicable data use agreement and credentialing requirements.

In particular, this repository is intended to support a restricted-data workflow in which:
- authorized users access the dataset through their own approved PhysioNet credentials
- the dataset is stored privately and is not redistributed through GitHub
- patient-level prediction files, row-level outputs, and any other restricted derivatives are not made public
- only aggregate results, summary tables, figures, and code are shared publicly

If collaborators wish to run the code, they should use their own authorized data access rather than shared access to a private dataset folder.

Additional guidance is provided in [`DATA_ACCESS.md`](DATA_ACCESS.md).

## Expected Dataset Files

Place the following files inside `data/raw/` for local execution, or in the corresponding Google Drive folder for Colab workflows:

- `echonext_metadata_100k.csv`
- `EchoNext_train_waveforms.npy`
- `EchoNext_val_waveforms.npy`
- `EchoNext_test_waveforms.npy`
- `EchoNext_no_split_waveforms.npy`
- `EchoNext_train_tabular_features.npy`
- `EchoNext_val_tabular_features.npy`
- `EchoNext_test_tabular_features.npy`
- `EchoNext_no_split_tabular_features.npy`

The first version of the benchmark uses only the `train`, `val`, and `test` splits. The `no_split` files are ignored in model training and evaluation.

## Directory Layout

```text
echonext_single_lead/
├── README.md
├── requirements.txt
├── configs/
├── data/
│   ├── raw/
│   └── processed/
├── notebooks/
├── outputs/
│   ├── figures/
│   ├── models/
│   ├── predictions/
│   └── tables/
├── scripts/
└── src/
```

## Installation

Python 3.10+ is recommended.

```bash
cd echonext_single_lead
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Reproducibility Notes

- The metadata `split` column is respected exactly as provided in the public dataset.
- Row order is preserved within each split so that metadata remains aligned with the waveform and tabular arrays.
- Waveforms are loaded with NumPy memory mapping to avoid loading the entire dataset into memory.
- Validation and test sets remain fixed across experiments.
- Each training run saves model checkpoints, prediction CSVs, and structured metrics files.
- Aggregate tables and figures are generated from saved outputs rather than by retraining models.
- Public releases of this repository should include code and aggregate artifacts only; restricted data files and patient-level outputs should remain private.

## Local Script Workflow

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
python scripts/04_run_lead_sweep.py --data_dir data/raw --label shd_moderate_or_greater_flag
```

Run the label-efficiency experiment:

```bash
python scripts/05_run_label_efficiency.py --data_dir data/raw --label shd_moderate_or_greater_flag
```

Run repeated-seed stability analysis for the core publication models:

```bash
python scripts/07_run_seed_stability.py --data_dir data/raw --label shd_moderate_or_greater_flag --seeds 42 43 44 --include_six_limb
```

Build publication-style summaries from the completed seeded runs:

```bash
python scripts/08_make_publication_results.py --label shd_moderate_or_greater_flag --seeds 42 43 44 --include_six_limb
```

Build aggregate benchmark tables and figures from completed runs:

```bash
python scripts/06_make_results_tables.py --output_dir outputs
```

## Colab Workflows

The repository includes three Colab notebooks:

- `notebooks/01_colab_starter.ipynb`
  - small proof-of-concept workflow
  - useful for smoke tests and first runs

- `notebooks/02_publication_pipeline.ipynb`
  - publication-style primary analysis workflow
  - runs the composite-SHD lead sweep, repeated-seed core comparison, summary tables, and figures

- `notebooks/03_secondary_label_pipeline.ipynb`
  - workflow for secondary-label analyses
  - supports optional full lead sweeps and repeated-seed core comparisons for labels such as `lvef_lte_45_flag` and `rv_systolic_dysfunction_moderate_or_greater_flag`

Each notebook assumes:
- code is cloned from GitHub into `/content/echonext_single_lead`
- the EchoNext dataset is stored in Google Drive
- outputs are copied back to Google Drive at the end of the run

## Output Files

Each training run creates a dedicated model directory under `outputs/models/` and corresponding prediction files under `outputs/predictions/`.

Per-run outputs include:
- `best_model.pt`
- `training_log.csv`
- `val_predictions.csv`
- `test_predictions.csv`
- `metrics.json`

Note that `test_predictions.csv` and `val_predictions.csv` are patient-level derived outputs and should be treated as restricted artifacts when generated from restricted data. They are appropriate for private analysis workflows but should not be committed to a public repository or shared publicly unless explicitly permitted by the governing data use agreement.

Aggregate and publication-oriented outputs may include:
- `outputs/tables/model_performance_by_input.csv`
- `outputs/tables/label_efficiency_results.csv`
- `outputs/tables/seed_stability_results.csv`
- `outputs/tables/seed_stability_summary.csv`
- `outputs/tables/publication_core_results.csv`
- `outputs/figures/auroc_by_input.png`
- `outputs/figures/auprc_by_input.png`
- `outputs/figures/label_efficiency_auroc.png`
- `outputs/figures/calibration_plot.png`
- `outputs/figures/seed_stability_auroc.png`
- `outputs/figures/publication_core_roc.png`
- `outputs/figures/publication_core_pr.png`
- `outputs/figures/publication_core_auroc_ci.png`
- `outputs/figures/publication_core_auprc_ci.png`

## Suggested Reproduction Order

For a full primary analysis reproduction:

1. Run `notebooks/02_publication_pipeline.ipynb`
2. Review `outputs/tables/publication_core_results.csv`
3. Review the benchmark figures under `outputs/figures/`

For secondary-label analyses:

1. Run `notebooks/03_secondary_label_pipeline.ipynb`
2. Change the `LABEL` variable to the desired outcome
3. Decide whether `RUN_FULL_SWEEP` should be `True` or `False`

## Reporting Standard

The project is written to support transparent reporting of prediction model development and evaluation and is intended to be described in line with TRIPOD-AI principles, including clear specification of the data source, predictors, outcomes, model development workflow, and performance assessment.

## Citation and Data Access

If you use this repository, please cite:
- the associated manuscript, once available
- the EchoNext dataset publication

Users should obtain the dataset directly from the official EchoNext/PhysioNet source and comply with the dataset license and usage terms.
