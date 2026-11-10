"""Nicheformer 封装(对比编码器)。

权重:HF `theislab/Nicheformer`,下载到 data/models/Nicheformer/;
平台均值 `technology_means/*_mean_script.npy` 来自 GitHub 仓库 data/model_means/。

与官方 tokenization 等价(theislab/nicheformer notebooks/tokenization/*):
1. 基因对齐到参考词表(model.h5ad 的 20310 个人类 Ensembl ID;小鼠基因先映射到人类直系同源);
2. 每个细胞归一化到 1e4,再除以平台均值;
3. 按表达降序取非零基因的索引 + 30 作为 token,前置 [物种, 平台, 模态] 三个特殊 token;
4. 取最后一层、去掉前 3 个特殊 token 后的均值作为细胞嵌入(512 维)。

注意:HF 自带 tokenizer 在本地路径下取不到参考基因表时只打印警告、静默跳过对齐,
所以这里自己做对齐,不使用它。
"""

from __future__ import annotations

from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp

AUX_TOKENS = 30
CONTEXT_LENGTH = 1500
MODALITY = {"spatial": 4, "dissociated": 3}
SPECIES = {"human": 5, "mouse": 6}
ASSAY = {"merfish": 7, "cosmx": 8, "xenium": 9, "10x_3_v3": 11}

# 平台 -> (assay token, 均值文件)。visium / starmap 不在预训练平台里,用最接近的近似。
PLATFORM = {
    "cosmx": ("cosmx", "cosmx"),
    "merfish": ("merfish", "merfish"),
    "xenium": ("xenium", "xenium"),
    "starmap": ("merfish", "iss"),  # 近似:原位测序类,无专属 token
    "visium": ("10x_3_v3", "dissociated"),  # 近似:spot 不是单细胞
}
APPROXIMATE_PLATFORMS = {"starmap", "visium"}


# ---------------------------------------------------------------- 基因映射

def human_symbol_to_ensembl(hgnc_path: Path | str) -> dict[str, str]:
    """HGNC 全表:当前 symbol 优先,其次旧 symbol 和别名(只取无歧义的)。"""
    hg = pd.read_csv(hgnc_path, sep="\t", dtype=str, usecols=[
        "symbol", "prev_symbol", "alias_symbol", "ensembl_gene_id"]).dropna(subset=["ensembl_gene_id"])
    mapping = dict(zip(hg["symbol"], hg["ensembl_gene_id"]))
    for col in ("prev_symbol", "alias_symbol"):
        ex = hg[["ensembl_gene_id", col]].dropna().assign(s=lambda d: d[col].str.split("|")).explode("s")
        ex = ex[~ex["s"].isin(mapping)]
        uniq = ex.groupby("s")["ensembl_gene_id"].nunique()
        ex = ex[ex["s"].isin(uniq[uniq == 1].index)].drop_duplicates("s")
        mapping.update(dict(zip(ex["s"], ex["ensembl_gene_id"])))
    return mapping


def mouse_symbol_to_human_ensembl(mgi_path: Path | str, hgnc_path: Path | str) -> dict[str, str]:
    """MGI HOM_MouseHumanSequence:同一同源组里 1 对 1 的 mouse/human 对,经 HGNC ID 转 Ensembl。"""
    hom = pd.read_csv(mgi_path, sep="\t", dtype=str)
    key = "DB Class Key"
    mouse = hom[hom["NCBI Taxon ID"] == "10090"][[key, "Symbol"]]
    human = hom[hom["NCBI Taxon ID"] == "9606"][[key, "HGNC ID"]].dropna()
    pairs = mouse.merge(human, on=key)
    pairs = pairs[pairs.groupby(key)[key].transform("size") == 1]  # 只保留 1:1 直系同源
    hg = pd.read_csv(hgnc_path, sep="\t", dtype=str, usecols=["hgnc_id", "ensembl_gene_id"]).dropna()
    pairs = pairs.merge(hg, left_on="HGNC ID", right_on="hgnc_id")
    return dict(zip(pairs["Symbol"], pairs["ensembl_gene_id"]))


def align_to_reference(adata: ad.AnnData, reference_genes: pd.Index, mapping: dict[str, str] | None,
                       ensembl_col: str | None = None) -> tuple[sp.csr_matrix, dict]:
    """把表达矩阵按参考词表顺序重排(cells × 20310),未覆盖的基因填 0。"""
    if ensembl_col is not None and ensembl_col in adata.var:
        ids = adata.var[ensembl_col].astype(str).values
    else:
        ids = np.array([mapping.get(g) for g in adata.var_names], dtype=object)
    ref_pos = pd.Series(np.arange(len(reference_genes)), index=reference_genes)
    ref_pos = ref_pos[~ref_pos.index.duplicated()]
    src, dst, seen = [], [], set()
    for j, gid in enumerate(ids):
        if gid is not None and gid in ref_pos.index and gid not in seen:
            src.append(j)
            dst.append(ref_pos[gid])
            seen.add(gid)
    X = sp.csr_matrix(adata.X)[:, src]
    P = sp.csr_matrix((np.ones(len(src)), (np.arange(len(src)), dst)),
                      shape=(len(src), len(reference_genes)))
    stats = {"n_genes": adata.n_vars, "n_mapped": len(src), "frac_mapped": len(src) / adata.n_vars}
    return (X @ P).tocsr().astype(np.float32), stats


# ---------------------------------------------------------------- tokenization

def tokenize(X: sp.csr_matrix, technology_mean: np.ndarray, species: str, platform: str,
             modality: str = "spatial", context_length: int = CONTEXT_LENGTH,
             nan_mode: str = "official", chunk: int = 1000) -> np.ndarray:
    """返回 (n_cells, context_length) 的 int64 token 矩阵,0 为 PAD。

    平台均值对该平台未测的基因是 NaN。
    - nan_mode="official":与官方 NicheformerDataset._process_chunk 完全一致。x / NaN = NaN,
      np.nonzero 把 NaN 当非零、argsort 把 NaN 排最后,于是每个细胞在真实基因之后
      都接上同一串"均值为 NaN 的基因",直到填满上下文。预训练时模型看到的就是这样的输入。
    - nan_mode="drop":丢弃均值为 NaN 的基因,剩余位置为 PAD(非官方,仅作对照)。
    """
    assay, _ = PLATFORM[platform]
    n_gene_tokens = context_length - 3
    mean = technology_mean.copy()
    mean += mean == 0
    if nan_mode == "drop":
        mean = np.where(np.isnan(mean), np.inf, mean)  # x / inf = 0 -> 不会成为 token

    tokens = np.zeros((X.shape[0], context_length), dtype=np.int64)
    tokens[:, 0] = SPECIES[species]
    tokens[:, 1] = ASSAY[assay]
    tokens[:, 2] = MODALITY[modality]
    for s in range(0, X.shape[0], chunk):
        x = np.nan_to_num(X[s:s + chunk].toarray().astype(np.float64))
        counts = x.sum(1, keepdims=True)
        counts += counts == 0
        x = x * (1e4 / counts) / mean.reshape(1, -1)
        for i, cell in enumerate(x):  # 与官方 _sub_tokenize_data 相同的 numpy 操作
            nz = np.nonzero(cell)[0]
            order = nz[np.argsort(-cell[nz])][:n_gene_tokens]
            tokens[s + i, 3:3 + len(order)] = order + AUX_TOKENS
    return tokens


# ---------------------------------------------------------------- 模型

class NicheformerEncoder:
    def __init__(self, model_dir: Path | str = "data/models/Nicheformer",
                 gene_map_dir: Path | str = "data/raw/gene_maps", device: str = "cuda"):
        import torch
        from transformers import AutoModelForMaskedLM

        self.model_dir, self.gene_map_dir, self.device = Path(model_dir), Path(gene_map_dir), device
        self.model = AutoModelForMaskedLM.from_pretrained(
            self.model_dir, trust_remote_code=True, dtype=torch.float32).to(device).eval()
        self.reference_genes = ad.read_h5ad(self.model_dir / "model.h5ad").var_names
        self._maps: dict[str, dict[str, str]] = {}

    def gene_map(self, species: str) -> dict[str, str]:
        if species not in self._maps:
            hgnc = self.gene_map_dir / "hgnc_complete_set.txt"
            self._maps[species] = (human_symbol_to_ensembl(hgnc) if species == "human" else
                                   mouse_symbol_to_human_ensembl(
                                       self.gene_map_dir / "HOM_MouseHumanSequence.rpt", hgnc))
        return self._maps[species]

    def technology_mean(self, platform: str) -> np.ndarray:
        _, mean_name = PLATFORM[platform]
        return np.load(self.model_dir / "technology_means" / f"{mean_name}_mean_script.npy")

    def embed(self, adata: ad.AnnData, species: str, platform: str,
              batch_size: int = 32, layer: int = -1,
              nan_mode: str = "official") -> tuple[np.ndarray, dict]:
        """返回 (n_cells, 512) 嵌入和基因覆盖率统计。adata.X 应为原始计数。"""
        import torch

        X, stats = align_to_reference(adata, self.reference_genes, self.gene_map(species),
                                      ensembl_col="gene_ids" if species == "human" else None)
        mean = self.technology_mean(platform)
        tokens = tokenize(X, mean, species, platform, nan_mode=nan_mode)
        stats["median_tokens"] = float(np.median((tokens[:, 3:] > 0).sum(1)))
        expressed = X.multiply(~np.isnan(mean).reshape(1, -1)).tocsr()
        stats["median_informative_tokens"] = float(np.median(np.diff((expressed > 0).tocsr().indptr)))
        stats["approximate_platform"] = platform in APPROXIMATE_PLATFORMS

        out = []
        with torch.no_grad():
            for s in range(0, len(tokens), batch_size):
                ids = torch.from_numpy(tokens[s:s + batch_size]).to(self.device)
                emb = self.model.get_embeddings(input_ids=ids, attention_mask=ids != 0,
                                                layer=layer, with_context=False)
                out.append(emb.float().cpu().numpy())
        return np.concatenate(out), stats
