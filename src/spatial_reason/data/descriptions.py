"""从数据自动生成邻域文字描述(细胞类型占比、高表达基因),用于阶段 0 对齐与文本基线 prompt。

描述只用细胞类型和表达,不含 niche / domain 标签,避免把下游任务的标签泄漏进对齐阶段。

- 组成:中心 + k 个邻居中占比 >= min_frac 的细胞类型(最多 max_types 个),百分比取整到 5%;
- 基因:邻域平均 log 表达相对整张切片的效应量 (mean_nb - mean_slice) / sd_slice 最大的基因,
  要求效应量 >= min_effect 且邻域内至少 min_detect 比例的细胞检出;
- 细胞类型名做轻度规范化(合并编号亚群,如 "Endothelial 2" -> "endothelial cells";
  CosMx 的 "tumor 9" 等病人特异性肿瘤簇合并为 "tumor cells")。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

import anndata as ad
import numpy as np
import scipy.sparse as sp

EXCLUDE_GENE = re.compile(r"^(Blank|NegPrb|NegControl|SystemControl|BLANK)", re.I)

STARMAP_NAMES = {
    "Astro": "astrocytes", "Endo": "endothelial cells", "Oligo": "oligodendrocytes",
    "Smc": "smooth muscle cells", "Micro": "microglia", "Lhx6": "Lhx6+ interneurons",
    "NPY": "NPY+ interneurons", "Reln": "Reln+ interneurons", "SST": "SST+ interneurons",
    "VIP": "VIP+ interneurons", "PVALB": "PVALB+ interneurons", "eL2/3": "L2/3 excitatory neurons",
    "eL4": "L4 excitatory neurons", "L5-1": "L5 excitatory neurons", "eL5-2": "L5 excitatory neurons",
    "eL5-3": "L5 excitatory neurons", "eL6-1": "L6 excitatory neurons", "eL6-2": "L6 excitatory neurons",
}
COSMX_NAMES = {
    "B-cell": "B cells", "NK": "NK cells", "T CD4 memory": "memory CD4 T cells",
    "T CD4 naive": "naive CD4 T cells", "T CD8 memory": "memory CD8 T cells",
    "T CD8 naive": "naive CD8 T cells", "Treg": "regulatory T cells", "endothelial": "endothelial cells",
    "epithelial": "epithelial cells", "fibroblast": "fibroblasts", "mDC": "myeloid dendritic cells",
    "macrophage": "macrophages", "mast": "mast cells", "monocyte": "monocytes",
    "neutrophil": "neutrophils", "pDC": "plasmacytoid dendritic cells", "plasmablast": "plasmablasts",
}
MERFISH_NAMES = {
    "Astrocyte": "astrocytes", "Ependymal": "ependymal cells", "Excitatory": "excitatory neurons",
    "Inhibitory": "inhibitory neurons", "Microglia": "microglia", "Pericytes": "pericytes",
    "Endothelial": "endothelial cells", "OD Mature": "mature oligodendrocytes",
    "OD Immature": "immature oligodendrocytes", "Ambiguous": "unassigned cells",
}


def display_name(cell_type: str, dataset: str) -> str:
    if dataset == "cosmx_nsclc":
        if cell_type.startswith("tumor"):
            return "tumor cells"
        return COSMX_NAMES.get(cell_type, cell_type)
    if dataset == "starmap":
        return STARMAP_NAMES.get(cell_type, cell_type)
    if dataset == "merfish_hypothalamus":
        base = re.sub(r"\s+\d+$", "", cell_type)
        return MERFISH_NAMES.get(base, base)
    return cell_type


TEMPLATES = [
    "The center cell is one of the {center}. The neighborhood consists of {comp}. {genes}",
    "This neighborhood is made up of {comp}, and the center cell belongs to the {center}. {genes}",
    "Composition: {comp}. Center cell: {center_s}. {genes}",
    "Around a center cell from the {center}, the neighbors are {comp}. {genes}",
]
GENE_TEMPLATES = [
    "{g} {are} highly expressed relative to the rest of the tissue.",
    "Enriched genes: {g}.",
    "Compared with the whole section, {g} {are} elevated here.",
    "Marker genes up in this neighborhood: {g}.",
]
NO_GENE = [
    "No gene stands out relative to the rest of the tissue.",
    "Enriched genes: none.",
    "No gene is clearly elevated compared with the whole section.",
    "Marker genes up in this neighborhood: none.",
]


def _join(items: list[str]) -> str:
    if len(items) <= 1:
        return "".join(items)
    return ", ".join(items[:-1]) + " and " + items[-1]


def _singular(name: str) -> str:
    return re.sub(r"s$", "", name) if not name.endswith("ss") else name


@dataclass
class Description:
    text: str
    cell_types: list[tuple[str, int]] = field(default_factory=list)  # (名称, 百分比)
    genes: list[str] = field(default_factory=list)
    center: str | None = None


class DescriptionBuilder:
    def __init__(self, adata: ad.AnnData, dataset: str, min_frac: float = 0.1, max_types: int = 4,
                 n_genes: int = 3, min_effect: float = 0.5, min_detect: float = 0.3):
        self.dataset = dataset
        self.min_frac, self.max_types = min_frac, max_types
        self.n_genes, self.min_effect, self.min_detect = n_genes, min_effect, min_detect

        ct = adata.obs["cell_type"]
        self.has_cell_type = ct.notna().any()
        if self.has_cell_type:
            names = np.array([display_name(str(c), dataset) if isinstance(c, str) else "unassigned cells"
                              for c in ct.astype(object)])
            self.type_names, self.type_codes = np.unique(names, return_inverse=True)

        keep = np.array([not EXCLUDE_GENE.match(g) for g in adata.var_names])
        self.genes = np.asarray(adata.var_names)[keep]
        X = sp.csr_matrix(adata.X)[:, keep].astype(np.float32)
        lib = np.asarray(X.sum(1)).ravel()
        lib[lib == 0] = 1.0
        X = sp.diags(1e4 / lib) @ X
        X.data = np.log1p(X.data)
        self.X = X.tocsr()
        mean = np.asarray(self.X.mean(0)).ravel()
        sq = np.asarray(self.X.multiply(self.X).mean(0)).ravel()
        self.mu = mean
        self.sd = np.sqrt(np.maximum(sq - mean ** 2, 1e-12))

    def composition(self, idx: np.ndarray) -> list[tuple[str, int]]:
        counts = np.bincount(self.type_codes[idx], minlength=len(self.type_names))
        frac = counts / len(idx)
        order = np.argsort(-frac, kind="stable")
        out = []
        for j in order[: self.max_types]:
            if frac[j] < self.min_frac:
                break
            pct = int(5 * round(100 * frac[j] / 5))
            out.append((str(self.type_names[j]), max(pct, 5)))
        return out

    def enriched_genes(self, idx: np.ndarray) -> list[str]:
        sub = self.X[idx]
        m = np.asarray(sub.mean(0)).ravel()
        detect = np.asarray((sub > 0).mean(0)).ravel()
        effect = (m - self.mu) / self.sd
        effect[(detect < self.min_detect) | (m <= self.mu)] = -np.inf
        top = np.argsort(-effect)[: self.n_genes]
        return [str(self.genes[j]) for j in top if effect[j] >= self.min_effect]

    def describe(self, idx: np.ndarray, rng: np.random.Generator | None = None,
                 template: int | None = None) -> Description:
        """idx:邻域细胞下标,第 0 个是中心细胞。"""
        t = template if template is not None else int((rng or np.random.default_rng()).integers(len(TEMPLATES)))
        genes = self.enriched_genes(idx)
        if genes:
            gtxt = GENE_TEMPLATES[t].format(g=_join(genes), are="is" if len(genes) == 1 else "are")
        else:
            gtxt = NO_GENE[t]

        if not self.has_cell_type:  # Visium:只有表达
            return Description(text=gtxt, genes=genes)

        comp = self.composition(idx)
        center = str(self.type_names[self.type_codes[idx[0]]])
        comp_txt = _join([f"{p}% {n}" for n, p in comp]) if comp else "a mixture of many cell types"
        text = TEMPLATES[t].format(center=center, center_s=_singular(center), comp=comp_txt, genes=gtxt)
        return Description(text=text, cell_types=comp, genes=genes, center=center)
