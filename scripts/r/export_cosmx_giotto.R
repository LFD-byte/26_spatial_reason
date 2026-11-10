# 从 NanoString 的 Giotto 对象导出 CosMx NSCLC 的细胞注释(cell_type / niche)和坐标。
# 不需要安装 Giotto:直接读 S4 对象的属性。
#
#   Rscript scripts/r/export_cosmx_giotto.R data/raw/cosmx_nsclc
#
# 输入: <root>/giotto/Processed Data Giotto Object/SMI_Giotto_Object.RData
# 输出: <root>/cell_annotations.csv

args <- commandArgs(trailingOnly = TRUE)
root <- if (length(args) >= 1) args[1] else "data/raw/cosmx_nsclc"
rdata <- file.path(root, "giotto", "Processed Data Giotto Object", "SMI_Giotto_Object.RData")

e <- new.env()
load(rdata, envir = e)
a <- attributes(e$gem)

md <- as.data.frame(a$cell_metadata$rna)
loc <- as.data.frame(a$spatial_locs$raw)

keep <- c("cell_ID", "fov", "slide_ID_numeric", "Run_Tissue_name", "Slide_name", "patient",
          "Area", "totalcounts", "remove_flagged_cells", "cell_type", "niche",
          "prop_tumor_in_100_neighbors")
out <- merge(md[, keep], loc, by = "cell_ID", sort = FALSE)
write.csv(out, file.path(root, "cell_annotations.csv"), row.names = FALSE)
cat("wrote", nrow(out), "cells\n")
