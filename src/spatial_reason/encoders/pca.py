"""PCA 编码器:log 归一化表达的 PCA(每细胞 64 维,不含空间上下文;邻域由 33 个细胞 token 提供)。

线性探针上它在三个数据集都优于 Novae(零样本与微调)和 Nicheformer,见 scripts/09_probe_encoders.py。

- 预处理:整数计数做 normalize_total(1e4) + log1p;MERFISH 的 X 已按细胞体积归一化(非整数),只做 log1p。
- PCA 只在训练切片上拟合(每张切片最多抽 5 万个细胞),验证切片只做变换,不参与拟合。
- 拟合结果保存在 data/models/pca/{dataset}.npz,之后各切片用同一个变换。
"""

from __future__ import annotations

import json
from pathlib import Path

import anndata as ad
import numpy as np

from spatial_reason.encoders.base import SpatialEncoder, register

PCA_ROOT = Path("data/models/pca")
DEFAULT_SPLIT_CONFIG = "configs/train/stage0_align.yaml"


def log_expr(adata: ad.AnnData) -> np.ndarray:
    import scanpy as sc
    import scipy.sparse as sp

    a = ad.AnnData(X=adata.X.copy())
    vals = a.X.data if sp.issparse(a.X) else np.asarray(a.X).ravel()
    if np.allclose(vals[:100000] % 1, 0):
        sc.pp.normalize_total(a, target_sum=1e4)
    sc.pp.log1p(a)
    X = a.X
    return np.asarray(X.toarray() if sp.issparse(X) else X, dtype=np.float32)


def train_samples(dataset: str, config: str = DEFAULT_SPLIT_CONFIG) -> list[str]:
    from spatial_reason.data.splits import split_samples
    from spatial_reason.utils.config import load_config

    datasets = load_config(config)["data"]["datasets"]
    if dataset in datasets:
        val = datasets[dataset]["val_samples"]
    else:  # 不在训练配置里的数据集(如 DLPFC):用 configs/data/{dataset}.yaml 的划分
        val = load_config(f"configs/data/{dataset}.yaml")["val_samples"]
    return split_samples(dataset, val)["train"]


@register("pca")
class PCAEncoder(SpatialEncoder):
    def __init__(self, n_components: int = 64, per_slice: int = 50000, seed: int = 0,
                 split_config: str = DEFAULT_SPLIT_CONFIG):
        self.n, self.per_slice, self.seed, self.split_config = n_components, per_slice, seed, split_config
        self._fits: dict[str, dict] = {}

    def _fit(self, dataset: str) -> dict:
        if dataset in self._fits:
            return self._fits[dataset]
        path = PCA_ROOT / f"{dataset}.npz"
        if not path.exists():
            from sklearn.decomposition import PCA

            from spatial_reason.data.datasets import load_sample

            rng = np.random.default_rng(self.seed)
            samples = train_samples(dataset, self.split_config)
            parts = []
            for s in samples:
                L = log_expr(load_sample(dataset, s))
                parts.append(L[rng.choice(len(L), min(len(L), self.per_slice), replace=False)])
            X = np.concatenate(parts)
            pca = PCA(n_components=min(self.n, X.shape[1] - 1), random_state=self.seed).fit(X)
            path.parent.mkdir(parents=True, exist_ok=True)
            np.savez(path, mean=pca.mean_, components=pca.components_,
                     explained=pca.explained_variance_ratio_, train_samples=np.array(samples))
            print(f"[pca] {dataset}: fit on {samples} ({len(X)} cells), "
                  f"explained {pca.explained_variance_ratio_.sum():.3f} -> {path}")
        d = np.load(path)
        self._fits[dataset] = {k: d[k] for k in d.files}
        return self._fits[dataset]

    def encode(self, adata: ad.AnnData, dataset: str) -> tuple[np.ndarray, dict]:
        f = self._fit(dataset)
        z = (log_expr(adata) - f["mean"]) @ f["components"].T
        return z.astype(np.float32), {"n_components": int(f["components"].shape[0]),
                                      "explained": float(f["explained"].sum()),
                                      "train_samples": f["train_samples"].tolist()}
