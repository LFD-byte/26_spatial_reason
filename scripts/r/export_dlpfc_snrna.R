# 从 spatialDLPFC snRNA-seq(Huuki-Myers et al. 2024,spatialLIBD::fetch_data("spatialDLPFC_snRNAseq"))
# 的 HDF5 SummarizedExperiment 中导出细胞注释和基因表。只用 base R:readRDS 不需要 Bioconductor 包,
# 通过属性直接取 S4 槽位。表达矩阵留在 assays.h5 里,由 Python 直接读。
#
#   Rscript scripts/r/export_dlpfc_snrna.R data/raw/dlpfc_snrna
args <- commandArgs(trailingOnly = TRUE)
root <- if (length(args)) args[1] else "data/raw/dlpfc_snrna"
x <- readRDS(file.path(root, "sce_DLPFC_annotated", "se.rds"))

flat <- function(ld) as.data.frame(lapply(ld, function(v) {
  if (is.factor(v)) as.character(v) else if (is.atomic(v)) v else NA
}), stringsAsFactors = FALSE)

cd <- attr(x, "colData")
cells <- flat(attr(cd, "listData"))
cells$cell_id <- attr(cd, "rownames")
write.csv(cells, file.path(root, "coldata.csv"), row.names = FALSE)

rr <- attr(x, "rowRanges")
genes <- flat(attr(attr(rr, "elementMetadata"), "listData"))
write.csv(genes, file.path(root, "rowdata.csv"), row.names = FALSE)
cat("cells:", nrow(cells), " genes:", nrow(genes), "\n")
