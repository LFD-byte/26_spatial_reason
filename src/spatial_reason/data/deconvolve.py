"""DLPFC Visium 的细胞类型反卷积:用 spatialDLPFC 单核参考,对每个 spot 做 NNLS,得到细胞类型比例。

与 LLMiniST 一致:参考数据来自同一研究(Huuki-Myers et al. 2024),细胞类型取 `cellType_layer`
(13 类,去掉 NA 和 Excit_ambig)。参考供体(10 人)与 12 张 Visium 切片的 3 位供体不重叠。

方法(轻量、CPU、可复现,思路同 RCTD 的平台效应校正):
1. 参考:每个细胞按文库大小归一到 1e4,按类型求平均,得到类型签名 S(基因 x 类型);
2. 标记基因:每个类型取 n_markers 个"该类型均值 / 其他类型最大均值"最高的蛋白编码基因
   (检出率 >= min_detect,且与 Visium 共有,按 Ensembl ID 匹配)。只用蛋白编码基因:单核数据里大量
   lncRNA 标记(核内转录本)在 Visium 上几乎测不到;
3. 平台效应:单核与 Visium 对同一基因的捕获效率不同。每个标记基因乘以系数
   g = Visium 全部 spot 的平均表达 / 参考的细胞加权平均表达;不做这一步时,一个低文库、
   以线粒体呼吸链基因为主的类型(Excit_L3/4/5)会在所有层吸走约 70% 的份额;
4. spot:同样归一到 1e4,在标记基因上解 NNLS(每个基因按签名最大值缩放,避免高表达基因主导),
   归一化后作为该 spot 的细胞类型组成。注意这是**转录本份额**,不是细胞数比例:按参考文库大小
   换算成细胞数会把神经元压得很低,层结构反而变模糊。

检查(12 张切片,按真值层平均):L4 以 Excit_L4 最高(18%),L5 以 Excit_L5 最高(14%),
L6 以 Excit_L5/6 与 Excit_L6 最高,WM 少突胶质细胞 55%,L1 星形胶质细胞最高。
"""

from __future__ import annotations

from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp

REF_ROOT = Path("data/raw/dlpfc_snrna")
PROPS_DIR = Path("data/processed/dlpfc_deconv")
TYPE_KEY = "cellType_layer"
DROP_TYPES = {"NA", "Excit_ambig", ""}


def build_signatures(root: Path = REF_ROOT, block: int = 1456) -> dict:
    """逐块读 assays.h5(稠密,细胞 x 基因),累加各类型的归一化表达与检出。"""
    import h5py

    cells = pd.read_csv(root / "coldata.csv", usecols=[TYPE_KEY, "sum"], keep_default_na=False)
    genes = pd.read_csv(root / "rowdata.csv", usecols=["gene_id", "gene_name", "gene_type"])
    types = np.array(cells[TYPE_KEY].astype(str))
    keep = ~np.isin(types, list(DROP_TYPES))
    names = sorted(set(types[keep]))
    code = np.full(len(types), -1)
    code[keep] = np.searchsorted(names, types[keep])

    T, G = len(names), len(genes)
    sums, detect = np.zeros((T, G)), np.zeros((T, G))
    libs, n = np.zeros(T), np.zeros(T)
    with h5py.File(root / "sce_DLPFC_annotated" / "assays.h5", "r") as f:
        X = f["assay001"]  # counts
        assert X.shape == (len(types), G), X.shape
        for s in range(0, X.shape[0], block):
            c = code[s:s + block]
            m = c >= 0
            if not m.any():
                continue
            B = X[s:s + block][m]
            lib = B.sum(1)
            Bn = B * (1e4 / np.maximum(lib, 1))[:, None]
            for t in np.unique(c[m]):
                r = c[m] == t
                sums[t] += Bn[r].sum(0)
                detect[t] += (B[r] > 0).sum(0)
                libs[t] += lib[r].sum()
                n[t] += r.sum()
            print(f"[ref] {min(s + block, X.shape[0])}/{X.shape[0]} cells", flush=True)
    return {"types": np.array(names), "gene_id": genes["gene_id"].to_numpy(dtype=str),
            "gene_name": genes["gene_name"].to_numpy(dtype=str),
            "gene_type": genes["gene_type"].to_numpy(dtype=str), "mean": sums / n[:, None],
            "detect": detect / n[:, None], "lib": libs / n, "n": n}


def select_markers(sig: dict, usable: np.ndarray, n_markers: int = 30, min_detect: float = 0.2,
                   ) -> np.ndarray:
    """usable:可用基因的布尔掩码(与 Visium 共有)。返回标记基因下标(去重)。"""
    M, D = sig["mean"], sig["detect"]
    usable = usable & (sig["gene_type"] == "protein_coding")
    out = []
    for t in range(len(sig["types"])):
        other = np.delete(M, t, axis=0).max(0)
        score = np.log2((M[t] + 1e-3) / (other + 1e-3))
        score[~usable | (D[t] < min_detect)] = -np.inf
        top = np.argsort(-score)[:n_markers]
        out.extend(top[np.isfinite(score[top])].tolist())
    return np.unique(out)


def _cp10k(adata: ad.AnnData, gene_ids: np.ndarray) -> np.ndarray:
    col = pd.Index(adata.var["gene_ids"].astype(str)).get_indexer(gene_ids)
    assert (col >= 0).all()
    X = sp.csr_matrix(adata.X).astype(np.float64)
    lib = np.asarray(X.sum(1)).ravel()
    return (sp.diags(1e4 / np.maximum(lib, 1)) @ X[:, col]).toarray()


def platform_factors(adatas: list[ad.AnnData], sig: dict, markers: np.ndarray) -> np.ndarray:
    """每个标记基因的 Visium / 单核 表达比(Visium 取全部切片 spot 的平均,参考按细胞数加权)。"""
    vis = np.concatenate([_cp10k(a, sig["gene_id"][markers]) for a in adatas]).mean(0)
    w = sig["n"] / sig["n"].sum()
    ref = (w[:, None] * sig["mean"][:, markers]).sum(0)
    return (vis + 1e-3) / (ref + 1e-3)


def deconvolve(adata: ad.AnnData, sig: dict, markers: np.ndarray, factors: np.ndarray) -> pd.DataFrame:
    """adata:Visium 原始计数,var 含 gene_ids。返回 spot x 类型的转录本份额(行和为 1)。"""
    from scipy.optimize import nnls

    Y = _cp10k(adata, sig["gene_id"][markers])  # spot x 标记基因
    S = sig["mean"][:, markers].T * factors[:, None]  # 标记基因 x 类型,已做平台校正
    scale = S.max(1, keepdims=True)
    S, Y = S / scale, Y / scale.T
    W = np.stack([nnls(S, y)[0] for y in Y])
    tot = W.sum(1, keepdims=True)
    P = np.where(tot > 0, W / np.maximum(tot, 1e-12), 1.0 / W.shape[1])
    # 类型名里的 "/"(如 Excit_L3/4/5)不能作 HDF5 键,改成 LLMiniST 的写法 Excit_L3_4_5
    cols = [str(t).replace("/", "_") for t in sig["types"]]
    return pd.DataFrame(P, index=adata.obs_names, columns=cols)


def attach_props(adata: ad.AnnData, sample: str, props_dir: Path | str = PROPS_DIR) -> bool:
    """若有反卷积结果,写入 obsm['cell_type_props'] 与 obs['cell_type'](主导类型)。"""
    from spatial_reason.data.descriptions import PROPS_KEY

    f = Path(props_dir) / f"{sample}.csv"
    if not f.exists():
        return False
    P = pd.read_csv(f, index_col=0).reindex(adata.obs_names)
    assert not P.isna().any().any(), f"{f}: 与 obs_names 不一致"
    adata.obsm[PROPS_KEY] = P
    adata.obs["cell_type"] = pd.Categorical(P.columns[P.to_numpy().argmax(1)])
    return True
