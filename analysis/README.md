# Analysis

## Contents

### Data Preprocessing Notebooks

| File | Dataset | Description |
|---|---|---|
| [`pbmc3k_prep.ipynb`](pbmc3k_prep.ipynb) | PBMC 3k (10x Genomics) | Quality-control filtering, normalisation, log-transformation, HVG selection, and cell-type annotation for the ~2,700-cell PBMC 3k dataset. |
| [`pbmc10k_prep.ipynb`](pbmc10k_prep.ipynb) | PBMC 10k (10x Genomics) | Same preprocessing pipeline applied to the larger ~10,000-cell PBMC 10k dataset; additionally exports HVG lists and CellTypist top-20 marker genes used by the runner. |
| [`zheng_prep.ipynb`](zheng_prep.ipynb) | Zheng 68k PBMCs | Preprocessing and subsampling for the large-scale Zheng 2017 68k-cell PBMC dataset. Relies on [`zheng_filter.R`](zheng_filter.R) and [`select_pure_pbmc.R`](select_pure_pbmc.R) for upstream filtering. |
| [`tian_prep.ipynb`](tian_prep.ipynb) | Tian 2019 (`sc_10x`, `sc_celseq2`, `sc_dropseq`) | Preprocessing for the three Tian 2019 protocol-comparison datasets used for cross-platform experiments. |
| [`Wu_Qian_prep.ipynb`](Wu_Qian_prep.ipynb) | Wu 2021 / Qian 2020 | Joint preprocessing notebook for the Wu 2021 (~24k cells) and Qian 2020 (~52k cells) breast-cancer / lung-cancer single-cell datasets. |
| [`synthetic_data_prep.ipynb`](synthetic_data_prep.ipynb) | SERGIO simulations | Loads, validates, and preprocesses SERGIO simulation outputs into the tabular format expected by the runner. |
| [`synthetic_data_generation.ipynb`](synthetic_data_generation.ipynb) | SERGIO simulations | Runs the SERGIO simulator across parameter sweeps to generate the synthetic scRNA-seq datasets used in the benchmark. |

### Figure & Analysis Notebooks

| File | Description |
|---|---|
| [`make_diff_plot.ipynb`](make_diff_plot.ipynb) | Generates differential analysis plots comparing feature-selection strategies across datasets and classifiers. |

### R Scripts

| File | Description |
|---|---|
| [`select_pure_pbmc.R`](select_pure_pbmc.R) | Curates pure cell-type populations from the Zheng 68k PBMC data (must be run before `zheng_prep.ipynb`). |
| [`zheng_filter.R`](zheng_filter.R) | Filters and downsamples the raw Zheng 68k count matrix to a manageable size. |
| [`util.R`](util.R) | Shared R utilities for normalisation, clustering, dimensionality reduction, and plotting used by the R scripts above. Contains code originally Copyright © 2016 10x Genomics, Inc. (Apache 2.0). |

### Standalone Python Script

| File | Description |
|---|---|
| [`make_geneset_figures.py`](make_geneset_figures.py) | Produces publication-quality gene-set performance figures (PR-AUC curves, ROC-AUC curves, Cleveland dot plots, lollipop/gap charts) from runner output CSVs. See usage below. |

---

## Recommended Preprocessing Order

For a full end-to-end run, prepare datasets in the following order:

```
1. R scripts (Zheng only)
   zheng_filter.R → select_pure_pbmc.R

2. Preprocessing notebooks (any order)
   pbmc3k_prep.ipynb
   pbmc10k_prep.ipynb
   zheng_prep.ipynb
   tian_prep.ipynb
   Wu_Qian_prep.ipynb

3. Synthetic data (if using SERGIO datasets)
   synthetic_data_generation.ipynb → synthetic_data_prep.ipynb
```

---

## Generating Figures

After running the benchmark pipeline, gene-set performance figures can be generated with:

```bash
# Edit the LEIDEN_FILES and CT_FILES path variables at the top of the script first, then:
python make_geneset_figures.py

# Optional: specify a custom output directory
python make_geneset_figures.py --out ./my_figures
```

Difference analysis plots are produced interactively via [`make_diff_plot.ipynb`](make_diff_plot.ipynb).

---

## Data

All raw input data must be obtained independently from the sources listed in the main [README](../README.md#datasets). No raw or processed data files are committed to this repository.
