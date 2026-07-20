# scRNA-seq Feature Selection Benchmark

Benchmarks feature-selection strategies (HVGs, housekeeping genes, mitochondrial genes, cluster-marker unions, top-k FS methods, random subsets) against sklearn/XGBoost classifiers on single-cell RNA-seq datasets (SERGIO simulations, Tian, Wu2021, Qian2020, Bischoff2021, pbmc10k, and others).

## Repository layout

```
.
├── runner_scRNA_Seq.py       # main entry point — runs whatever experiment(s) config.yaml enables
├── fs_utils.py                 # config loading/validation, model zoo, FS methods, eval + plotting helpers
├── run_benchmark_tmux.sh      # launches runner_scRNA_Seq.py inside a named, detachable tmux session
├── config.yaml                 # experiment configuration — path is hardcoded in fs_utils.load_config()
├── gene_metadata.parquet      # ENSEMBL <-> HUGO symbol lookup — read unconditionally, every run
├── requirements.txt
├── Plots/                      # output dir — created automatically on import
└── logs/                       # tmux session stdout+stderr logs — created automatically
```

Datasets live **outside** the repo, one level up:

```
../Data/
├── Sergio/
│   ├── <dataset>.csv
│   └── meta_<dataset>.csv          # required for any dataset name containing 'sergio' — holds the 'Label' column
├── pbmc10k_after_clustering/
│   ├── HVG1000_genes.csv
│   └── pbmc10k_celltypist_top20_genes.csv
├── Housekeeping_GenesHuman.csv
└── ...                              # Wu2021 / Qian2020 / Bischoff2021 (.parquet), NIPS_2003 (Arcene/Gisette/Madelon), sc_10x / sc_celseq2 / sc_dropseq, crc_GSE81861, etc.
```

`download_datasets` in `config.yaml` is `0`, so data must be placed here manually.

**Everything below assumes commands are run from the repo root** — 

`fs_utils.py` writes plots/CSVs to relative paths (`./Plots/...`), reads `config.yaml` and `gene_metadata.parquet` from the current directory, and `runner_scRNA_Seq.py` resolves `../Data/...` relative to it too. 

`run_benchmark_tmux.sh` already `cd`s into wherever it's invoked from, so just run it from the repo root.

## Setup

```bash
python3 -m venv fs_env
source fs_env/bin/activate
pip install -r requirements.txt
```

`tmux` also needs to be installed on the host (`sudo apt install tmux` / `brew install tmux`).

## Required files, checked before any modeling starts

`runner_scRNA_Seq.py` validates these four paths from `config.yaml` at startup and exits immediately with a clear `FileNotFoundError` if any is missing:

- `filepath` (the dataset)
- `hvg_file_path`
- `hk_genes_file_path`
- `union_cluster_genes_file_path`

## Configuring an experiment

Everything is driven by `config.yaml` (loaded once as `fs_utils.config`, validated on load — invalid `model_name`, non-string `dataset`/`target`, etc. fail fast with an `AssertionError`).

| Key | Purpose |
|---|---|
| `dataset`, `filepath` | which dataset to load and where from |
| `target` | label column name (`'type'` for most datasets) |
| `model_name` | `RF`, `LR`, `SVM`, `DT`, `NN`/`MLP`, `XGB`, `GBM`, `HistGB`, `Ridge`, `SGD`, or `ET` (Extra Trees) |
| `train_test_sep` | `1` to evaluate on a separate `test_dataset`/`test_filepath`, `0` to cross-validate/split internally |
| `use_only_hvgs` / `use_only_hk_genes` / `use_only_mt_genes` / `use_union_of_cluster_genes` / `use_uc_plus` / `use_only_non_hvgs` / `use_only_fs_list` | restrict to one specific gene subset |
| `run_with_all_features`, `run_var_random_subsets_of_all_features`, `run_random_subset_selected_features`, `run_perc_random_subset_selected_features`, `run_topk_fs_evaluation`, `run_random_train_test_split`, `run_null_model` | which experiment(s) actually execute |
| `num_runs` | repeats per subset size |
| `random_state` | seed |
| `interactive` | if `1`, some code paths call `input()` for a y/n confirmation — **leave this `0`** for unattended tmux runs, or the session will just hang waiting for stdin |

**Only one gene-subset flag may be `1` at a time** — `fs_utils.py` raises `ValueError: multiple gene-selection flags active` if more than one of `use_only_*`/`use_union_of_cluster_genes`/`use_uc_plus` is set. Leave them all `0` to use every gene.

To reproduce a specific prior result: edit the fields above in `config.yaml` to match that experiment (or check out the commit/branch where that `config.yaml` version lives), then launch with a tmux session name that describes it — **the session name is just a log-file label, it does not select the dataset or experiment.** What actually runs always comes from whatever `config.yaml` currently contains.

If `run_topk_fs_evaluation: 1`: only `lasso`, `elasticnet`, and `rf_importance` are active methods (see `FS_METHODS` in `fs_utils.py`) — evaluated over `k_list`.

## Running

```bash
chmod +x run_benchmark_tmux.sh   # first time only
./run_benchmark_tmux.sh <session_name>
```

Example, matching a SERGIO low-variance gene experiment:

```bash
./run_benchmark_tmux.sh 50_60_gene_low_var_sergio_genes_31_onwards_all_subsets
```

This starts (or re-attaches to, if the name already exists) a tmux session that activates `fs_env` and runs `python3 runner_scRNA_Seq.py`, teeing combined stdout+stderr to `logs/<session_name>_output.log`.

- **Watch live:** `tmux attach -t <session_name>` (detach again with `Ctrl+B`, then `D`)
- **Tail the log without attaching:** `tail -f logs/<session_name>_output.log`
- **Stop it:** `tmux kill-session -t <session_name>`

## Outputs

Everything lands under a path `fs_utils.py` builds automatically (and creates on import):

- `train_test_sep: 0` → `./Plots/<dataset>/<model_name>/<gene_subset_suffix?>/`
- `train_test_sep: 1` → `./Plots/<dataset>/test_<test_dataset>/<model_name>_train_test_sep/<gene_subset_suffix?>/`

(`<gene_subset_suffix?>` is one of `hvgs`, `non_hvgs`, `mt_genes`, `hk_genes`, `union_cluster_genes`, `uc_plus`, `fs_list` — only present if the matching flag is `1`.)

What each active toggle produces there, from the config included with this repo:

| Toggle | Output |
|---|---|
| `run_with_all_features` | printed accuracy/AUC — no file, just console/log |
| `run_var_random_subsets_of_all_features` | `<dataset>_<model>_(D/O)_random_features_all_extended_results.csv` (per-subset-size mean/median/std across metrics). Note: the corresponding plot call is commented out in `runner_scRNA_Seq.py`, so no PDF is produced for this toggle currently — only the CSV. |
| `run_random_train_test_split` | confusion-matrix PDF + `random_train_test_split_results_<dataset>.csv`, averaged over `num_runs` random splits |
| `run_perc_random_subset_selected_features` / `run_random_subset_selected_features` | extended-results CSV per subset size/percentage, plus confusion-matrix PDFs if `save_cm_subset_selected_features: 1` |
| `run_topk_fs_evaluation` | `fs_topk_summary.csv` |
| `get_sparsity_summary` | `<dataset>_sparsity_summary.csv` |

Console output also logs dataset shape, class balance, and per-experiment timing — check `logs/<session_name>_output.log` if a run needs auditing after the fact.
