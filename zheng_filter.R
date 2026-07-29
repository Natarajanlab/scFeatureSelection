library(Seurat)
library(Matrix)
library(ggplot2)
library(Rtsne)
library(svd)
library(dplyr)
library(data.table)
library(anndata)

source('util.R')
source('select_pure_pbmc.R')

meta <- readRDS('zheng_meta.rds')
data <-readRDS('zheng_pure_pbmc_data.rds')

pure_pbmcs <- readRDS('zheng_pure_pbmc_data.rds')
all_data <- pure_pbmcs$all_data
all_json <- pure_pbmcs$all_json
all_metrics <- pure_pbmcs$all_metrics
all_mol_info <- pure_pbmcs$all_mol_info
genes <- all_data[[1]]$hg19$gene_symbols

set.seed(1)
rpc <- all_metrics %>% 
  mutate(conf_mapped_rpc=raw_rpc*conf_mapped_frac*good_bc_frac*good_umi_frac) %>%
  select(sample_id, description, conf_mapped_rpc)
tgt_rpc <- floor(min(rpc$conf_mapped_rpc)) # 13995
subsampled_purified_mats <- lapply(1:length(all_data), function(i) { # subsample the matrix to match tgt_rpc
  cat(sprintf("%d...\n", i))
  .downsample_gene_bc_mtx(all_json[[i]], all_data[[i]], all_mol_info[[i]], tgt_rpc, 'conf_mapped_reads')[[1]]
} )
set.seed(1)
all_pure_pca<-lapply(1:length(subsampled_purified_mats),function(i) pure_pca_i<-.do_propack(subsampled_purified_mats[[i]],10))
all_pure_tsne<-lapply(1:length(all_pure_pca),function(i) pure_tsne_i<-Rtsne(all_pure_pca[[i]]$pca,pca=F))

pure_id<-c("CD34+","CD56+ NK","CD4+/CD45RA+/CD25- Naive T", "CD4+/CD25 T Reg","CD8+/CD45RA+ Naive Cytotoxic",
           "CD4+/CD45RO+ Memory","CD8+ Cytotoxic T","CD19+ B","CD4+ T Helper2","CD14+ Monocyte","Dendritic")
FIG_DIR = './zheng_pbmc/'
sub_idx <-list(data.frame(sample=1, use=(get_pure_pop_idx(genes,pure_id[1],all_pure_pca[[1]],all_pure_tsne[[1]],FIG_DIR))), 
               data.frame(sample=2, use=(get_pure_pop_idx(genes,pure_id[2],all_pure_pca[[2]],all_pure_tsne[[2]],FIG_DIR))),
               data.frame(sample=3, use=(get_pure_pop_idx(genes,pure_id[3],all_pure_pca[[3]],all_pure_tsne[[3]],FIG_DIR))),
               data.frame(sample=4, use=(get_pure_pop_idx(genes,pure_id[4],all_pure_pca[[4]],all_pure_tsne[[4]],FIG_DIR))),
               data.frame(sample=5, use=(get_pure_pop_idx(genes,pure_id[5],all_pure_pca[[5]],all_pure_tsne[[5]],FIG_DIR))),
               data.frame(sample=6, use=(get_pure_pop_idx(genes,pure_id[6],all_pure_pca[[6]],all_pure_tsne[[6]],FIG_DIR))),
               data.frame(sample=7, use=(get_pure_pop_idx(genes,pure_id[7],all_pure_pca[[7]],all_pure_tsne[[7]],FIG_DIR))),
               data.frame(sample=8, use=(get_pure_pop_idx(genes,pure_id[8],all_pure_pca[[8]],all_pure_tsne[[8]],FIG_DIR))),
               data.frame(sample=9, use=(get_pure_pop_idx(genes,pure_id[9],all_pure_pca[[9]],all_pure_tsne[[9]],FIG_DIR))),
               data.frame(sample=10,use=(get_pure_pop_idx(genes,pure_id[10],all_pure_pca[[10]],all_pure_tsne[[10]],FIG_DIR))),
               data.frame(sample=10,use=(get_pure_pop_idx(genes,pure_id[11],all_pure_pca[[10]],all_pure_tsne[[10]],FIG_DIR))))
pure_select_11<-lapply(1:length(sub_idx),function(i) {subsampled_purified_mats[[sub_idx[[i]]$sample[1]]][sub_idx[[i]]$use,]})

target_indices <- c(8, 10, 3, 5, 2)
target_names   <- c("B cells", "CD14+ Monocytes", "CD4+ Naive T cells", "CD8+ Naive T cells", "NK cells")
sampled_list <- lapply(seq_along(target_indices), function(i) {
  orig_idx <- target_indices[i]
  mat <- subsampled_purified_mats[[orig_idx]]
  
  n_available <- nrow(mat)
  # Take 2600 or the maximum available if less than 2600
  n_to_sample <- min(n_available, 2600)
  
  sample_rows <- sample(seq_len(n_available), size = n_to_sample)
  return(mat[sample_rows, , drop = FALSE])
})

combined_matrix <- do.call(rbind, sampled_list)
cell_types <- rep(target_names, sapply(sampled_list, nrow))
obs_data <- data.frame(
  cell_type = cell_types,
  row.names = rownames(combined_matrix)
)
adata <- AnnData(
  X = as(combined_matrix, "dgCMatrix"), # Ensure sparse format for efficiency
  obs = obs_data,
  var = data.frame(gene_symbols = genes)
)
adata$write_h5ad("zheng_pbmc_dubstepr.h5ad")






