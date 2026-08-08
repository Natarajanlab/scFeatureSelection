# scFeatureSelection: systematic evaluation of feature selection strategies reveals transcriptomic redundancy in single-cell RNA sequencing
Bhavesh Neekhra, Shreyansh Priyadarshi, Kedar Natarajan

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.9%2B-blue)](https://www.python.org/)
[![R](https://img.shields.io/badge/R-4.0%2B-276DC3)](https://www.r-project.org/)

> **Code accompanying manuscript ** 

---

## Overview

Single-cell RNA-seq (scRNA-seq) datasets routinely contain thousands of genes, yet only a small subset carry information useful for cell-type classification. This repository provides a comprehensive benchmarking framework that systematically evaluates the classification utility of several biologically motivated and data-driven gene-selection strategies across multiple real and synthetic scRNA-seq datasets.

**Gene subsets evaluated:**

| Strategy | Description |
|---|---|
| **All features** | Full gene matrix — upper-bound reference |
| **HVGs** | Highly variable genes (pre-computed with Scanpy/Seurat) |
| **Non-HVGs** | Complement of the HVG set |
| **Housekeeping genes** | Constitutively expressed genes (Eisenberg & Levanon list) |
| **Union-Cluster genes** | Union of top marker genes per cluster (CellTypist / Leiden) |
| **Random subsets** | Randomly sampled gene subsets of varying size (null baseline) |

**Classifiers benchmarked:** Random Forest (`RF`), Logistic Regression (`LR`), Linear SVM (`SVM`), Decision Tree (`DT`), MLP (`NN`), XGBoost (`XGB`), Gradient Boosting (`GBM`), Histogram GBM (`HistGB`), Ridge, SGD, Extra Trees (`ET`).

**Metrics:** Accuracy, balanced accuracy, F1 (macro), precision, recall, ROC-AUC, PR-AUC, MCC.

---

## Datasets

The framework has been validated on the following datasets (data must be obtained independently — see [Data Availability](#data-availability)):

| Dataset | Type | Approx. cells | Source |
|---|---|---|---|
| Synthetic scRNA-seq simulated data | Synthetic | Variable | [Dibaeinia & Sinha, 2020](https://doi.org/10.1016/j.cels.2020.08.003) |
| Tian 2019 (`sc_10x`, `sc_celseq2`, `sc_dropseq`) | Real | ~3,000 | [Tian et al., 2019](https://doi.org/10.1038/s41592-019-0425-8) |
| Wu 2021 | Real | ~24,000 | [Wu et al., 2021](https://doi.org/10.1038/s41588-021-00911-1) |
| Qian 2020 | Real | ~52,000 | [Qian et al., 2020](https://doi.org/10.1016/j.celrep.2020.108161) |
| PBMC 3k (10x Genomics) | Real | ~2,700 | [10x Genomics](https://www.10xgenomics.com/resources/datasets) |
| PBMC 10k (10x Genomics) | Real | ~10,000 | [10x Genomics](https://www.10xgenomics.com/resources/datasets) |

---

## Repository Layout

All files reside in the repository root.

```
scFeatureSelection/
│
├── runner_scRNA_Seq.py        # Main entry point — orchestrates all experiments
├── fs_utils.py                # Core library: config, model zoo, FS methods, eval & plotting
├── make_geneset_figures.py    # Standalone figure-generation script (gene-set performance plots)
│
├── config.yaml                # Experiment configuration (edit this to run experiments)
├── gene_metadata.parquet      # ENSEMBL ↔ HUGO symbol lookup table (read on every run)
├── requirements.txt           # Python dependencies
│
├── run_benchmark_tmux.sh      # Runs runner_scRNA_Seq.py in a detachable tmux session
│
├── select_pure_pbmc.R         # R: curates pure cell populations from Zheng 68k PBMC data
├── zheng_filter.R             # R: filters and subsamples the Zheng 68k PBMC matrix
├── util.R                     # R: shared utilities (normalisation, clustering, plotting)
│
├── Wu_Qian_prep.ipynb         # Notebook: preprocessing for Wu 2021 / Qian 2020
├── pbmc3k_prep.ipynb          # Notebook: preprocessing for PBMC 3k
├── pbmc10k_prep.ipynb         # Notebook: preprocessing for PBMC 10k
├── tian_prep.ipynb            # Notebook: preprocessing for Tian 2019
├── synthetic_data_prep.ipynb  # Notebook: SERGIO simulation loading & preprocessing
├── synthetic_data_generation.ipynb  # Notebook: SERGIO simulation parameter sweeps
├── make_diff_plot.ipynb       # Notebook: differential analysis plots
│
├── LICENSE                    # Apache 2.0
└── .gitignore
```

> **Data** lives **outside** the repository root, one level up:
> ```
> ../Data/
> ├── Sergio/
> │   ├── <dataset>.csv
> │   └── meta_<dataset>.csv       # required for any dataset with 'sergio' in its name
> ├── pbmc10k_after_clustering/
> │   ├── HVG1000_genes.csv
> │   └── pbmc10k_celltypist_top20_genes.csv
> ├── Housekeeping_GenesHuman.csv
> └── ...                          # other datasets (parquet / CSV)
> ```

---

## Installation

### Python environment

```bash
python -m venv fs_env
source fs_env/bin/activate          # Windows: fs_env\Scripts\activate
pip install -r requirements.txt
```

**Core Python dependencies:**

| Package | Version |
|---|---|
| `numpy` | 1.26.4 |
| `pandas` | 2.2.0 |
| `scikit-learn` | 1.5.2 |
| `xgboost` | 2.0.1 |
| `matplotlib` | 3.8.0 |
| `seaborn` | 0.13.2 |
| `PyYAML` | 6.0.2 |
| `pyarrow` | 15.0.0 |
| `tqdm` | 4.66.1 |

**Optional (feature selection methods):**
```bash
pip install boruta skrebate pyHSICLasso
```
If not installed, the respective methods are silently skipped.

### R environment (for PBMC / Zheng preprocessing only)

```r
install.packages(c("Seurat", "Matrix", "ggplot2", "Rtsne", "svd",
                   "dplyr", "data.table", "pheatmap"))
BiocManager::install("anndata")
```

### tmux (for background runs)

```bash
sudo apt install tmux   # Ubuntu/Debian
brew install tmux       # macOS
```

---

## Quick Start

### 1. Place datasets

Ensure all required data files exist at the paths specified in `config.yaml` (paths default to `../Data/`).

### 2. Configure the experiment

Edit `config.yaml` — the primary fields to set:

```yaml
dataset:   50_60_gene_high_var_sergio_genes_31_onwards   # dataset name
filepath:  ../Data/Sergio/50_60_gene_high_var_sergio_genes_31_onwards.csv
target:    type         # label column
model_name: RF          # classifier (RF, LR, SVM, DT, NN, XGB, GBM, HistGB, Ridge, SGD, ET)
num_runs:  20           # repetitions per subset size
random_state: 42
```

### 3. Validate required files

`runner_scRNA_Seq.py` checks all four required file paths at startup and exits immediately with a clear `FileNotFoundError` if any is missing:

- `filepath` (dataset)
- `hvg_file_path`
- `hk_genes_file_path`
- `union_cluster_genes_file_path`

### 4. Run

**Option A — tmux (recommended for long runs):**
```bash
chmod +x run_benchmark_tmux.sh   # first time only
./run_benchmark_tmux.sh <session_name>
```

**Option B — direct:**
```bash
source fs_env/bin/activate
python runner_scRNA_Seq.py
```

**Monitor / manage a tmux run:**
```bash
tmux attach -t <session_name>          # reattach
tail -f logs/<session_name>_output.log  # tail log without attaching
tmux kill-session -t <session_name>    # stop run
```

---

## Configuration Reference

`config.yaml` controls every aspect of an experiment. **Only one** gene-subset flag (`use_only_*` / `use_union_of_cluster_genes` / `use_uc_plus`) may be set to `1` at a time; leaving all at `0` uses every gene.

### Dataset

| Key | Type | Description |
|---|---|---|
| `dataset` | `str` | Dataset identifier (used for output naming) |
| `filepath` | `str` | Path to the main dataset file |
| `target` | `str` | Name of the label column (e.g. `'type'`) |
| `train_test_sep` | `0/1` | `1` = evaluate on a separate held-out set defined by `test_dataset` / `test_filepath`; `0` = internal split / cross-validation |
| `test_dataset`, `test_filepath` | `str` | Only required when `train_test_sep: 1` |
| `hvg_file_path` | `str` | CSV of pre-computed HVG gene names |
| `hk_genes_file_path` | `str` | CSV of housekeeping gene names |
| `union_cluster_genes_file_path` | `str` | CSV of union-of-cluster marker genes |

### Model

| Key | Type | Description |
|---|---|---|
| `model_name` | `str` | `RF`, `LR`, `SVM`, `DT`, `NN`/`MLP`, `XGB`, `GBM`, `HistGB`, `Ridge`, `SGD`, `ET` |
| `opt_model` | `0/1` | `1` uses optimised (grid-searched) hyperparameters |
| `random_state` | `int` | Global random seed |
| `num_runs` | `int` | Repetitions per condition |

### Gene-subset flags (mutually exclusive)

| Key | Description |
|---|---|
| `use_only_hvgs` | Restrict to HVGs |
| `use_only_non_hvgs` | Restrict to non-HVGs |
| `use_only_hk_genes` | Restrict to housekeeping genes |
| `use_only_mt_genes` | Restrict to mitochondrial genes |
| `use_union_of_cluster_genes` | Restrict to union-of-cluster marker genes |
| `use_uc_plus` | Union-cluster genes + additional HVG/FS genes |
| `use_only_fs_list` | Restrict to genes from an external FS list (`fs_list_file_path`) |

### Experiment toggles

| Key | Description |
|---|---|
| `run_with_all_features` | Evaluate classifier with the full feature set |
| `run_random_train_test_split` | Evaluate over `num_runs` random train/test splits |
| `run_var_random_subsets_of_all_features` | Sweep over random gene subsets of increasing size (`feature_ticks_ranges`) |
| `run_topk_fs_evaluation` | Evaluate Lasso / ElasticNet / RF-importance top-k subsets over `k_list` |
| `run_perc_random_subset_selected_features` | Random % subsets of the chosen gene set |
| `run_random_subset_selected_features` | Fixed-size random subsets of the chosen gene set |
| `run_null_model` | Include a majority-class null-model baseline |
| `get_sparsity_summary` | Compute and save dataset sparsity statistics |

### Sweep parameters

| Key | Default | Description |
|---|---|---|
| `feature_ticks_ranges` | `[[10, 301, 10]]` | `[start, stop, step]` ranges for absolute feature-count sweeps |
| `feature_ticks_perc_ranges` | `[[0.1, 1.0, 0.1], ...]` | Percentage sweep ranges |
| `k_list` | `[5, 10, 20, 50]` | Top-k values for FS evaluation |
| `remove_cols` | `false` | `false` = independent random sampling (with replacement); `true` = random partition (no overlap) |

### Output / display

| Key | Description |
|---|---|
| `interactive` | Set to `0` for all unattended / tmux runs — avoids blocking `input()` calls |
| `debug` | `1` enables verbose diagnostic output |
| `show_progress_bar` | `1` shows a progress spinner (TTY only) |
| `save_cm_subset_selected_features` | `1` saves per-subset confusion-matrix PDFs |

---

## Outputs

All outputs are written to an automatically created subdirectory of `./Plots/`:

| `train_test_sep` | Output path |
|---|---|
| `0` | `./Plots/<dataset>/<model_name>/[<gene_subset>/]` |
| `1` | `./Plots/<dataset>/test_<test_dataset>/<model_name>_train_test_sep/[<gene_subset>/]` |

`<gene_subset>` is one of `hvgs`, `non_hvgs`, `mt_genes`, `hk_genes`, `union_cluster_genes`, `uc_plus`, `fs_list` — present only when the corresponding flag is `1`.

| Toggle | Files produced |
|---|---|
| `run_with_all_features` | Console/log only (no file) |
| `run_random_train_test_split` | Confusion-matrix PDF + `random_train_test_split_results_<dataset>.csv` |
| `run_var_random_subsets_of_all_features` | `<dataset>_<model>_(D/O)_random_features_all_extended_results.csv` |
| `run_topk_fs_evaluation` | `fs_topk_summary.csv` |
| `run_perc_random_subset_selected_features` | Extended-results CSV per percentage size |
| `run_random_subset_selected_features` | Extended-results CSV per subset size |
| `get_sparsity_summary` | `<dataset>_sparsity_summary.csv` |

Console output additionally logs dataset shape, class balance, and per-experiment timing. Log files are saved to `logs/<session_name>_output.log` when using the tmux runner.

---

## Reproducing Paper Results

Each experiment in the paper maps to a specific `config.yaml` configuration. The general workflow is:

1. Obtain and place the relevant dataset at the path specified in `config.yaml`.
2. Set `dataset`, `filepath`, `target`, `model_name`, and the appropriate experiment toggle.
3. Set `num_runs: 20` and `random_state: 42` to match the paper's settings.
4. Run via `./run_benchmark_tmux.sh <descriptive_session_name>`.
5. Results appear in `./Plots/<dataset>/` and `logs/<session_name>_output.log`.

> **Example** — reproducing the SERGIO high-variance 50-gene experiment:
> ```yaml
> dataset: 50_60_gene_high_var_sergio_genes_31_onwards
> filepath: ../Data/Sergio/50_60_gene_high_var_sergio_genes_31_onwards.csv
> model_name: RF
> run_var_random_subsets_of_all_features: 1
> num_runs: 20
> random_state: 42
> ```
> ```bash
> ./run_benchmark_tmux.sh sergio_high_var_50gene_RF
> ```

### Figure generation

Publication-quality gene-set performance figures (PR-AUC curves, ROC-AUC curves, Cleveland dot plots, lollipop/gap charts) are produced by `make_geneset_figures.py`:

```bash
# Edit the LEIDEN_FILES and CT_FILES path variables at the top of the script, then:
python make_geneset_figures.py
python make_geneset_figures.py --out ./my_figures   # custom output directory
```

---

## Feature Selection Methods

When `run_topk_fs_evaluation: 1`, the following methods are applied:

| Method | Implementation | Notes |
|---|---|---|
| **Lasso** | `LogisticRegressionCV(penalty='l1', solver='saga')` | Falls back to `SGDClassifier(penalty='l1')` for `p > 10,000, n < 2,000` |
| **Elastic Net** | `ElasticNetCV` | Combines L1 + L2 penalties |
| **RF Importance** | `RandomForestClassifier` mean decrease in impurity | Aggregated across all trees |
| **Boruta** | `BorutaPy` | Optional — requires `pip install boruta` |
| **ReliefF** | `skrebate.ReliefF` | Optional — requires `pip install skrebate` |
| **HSIC Lasso** | `pyHSICLasso` | Optional — requires `pip install pyHSICLasso` |

Optional methods are automatically skipped if the corresponding package is not installed.

---

## Data Availability

Raw data for real datasets are publicly available from the sources listed in the [Datasets](#datasets) table. SERGIO simulation scripts are included in `synthetic_data_generation.ipynb`. Pre-processed files required at runtime (`../Data/`) must be generated using the per-dataset preprocessing notebooks included in this repository. No raw data files are committed to this repository.

---

## Citation

> Neekhra B, Priyadarshi S and Natarajan KN (2026) scFeatureSelection: systematic evaluation of feature selection strategies reveals transcriptomic redundancy in single-cell RNA sequencing

If you use this code in your research, please cite the associated manuscript.

---

## License

This project is licensed under the **Apache License 2.0** — see [LICENSE](LICENSE) for details.

The R utilities (`util.R`, `select_pure_pbmc.R`) contain code originally Copyright © 2016 10x Genomics, Inc., distributed under the same Apache 2.0 licence.
