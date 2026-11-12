"""表达输入(方案 B):不做降维,把每个细胞的标准化 log 表达放进所有数据集共享的基因空间,
由模型里的可学习基因集合编码器(models/projector.py: GeneSetEncoder)端到端编码。

- 基因表:各数据集基因名转大写后的并集(人 / 小鼠同名基因按大写对齐),保存在 data/models/expr/vocab.json,
  列顺序固定;某个 panel 没测的基因填 0。
- 每个基因:log 表达(与 PCA 相同的 normalize_total + log1p;MERFISH 只做 log1p)按该数据集
  训练切片的均值 / 标准差做 z-score,截断到 [-5, 10]。0 因此表示"与训练切片平均水平相同",
  和"没测"一样不贡献信息。统计量只来自训练切片。
- 以 float16 缓存(CosMx 约 77 万细胞 × 1210 基因)。
"""

from __future__ import annotations

import json
from pathlib import Path

import anndata as ad
import numpy as np

from spatial_reason.encoders.base import SpatialEncoder, register
from spatial_reason.encoders.pca import DEFAULT_SPLIT_CONFIG, log_expr, train_samples

EXPR_ROOT = Path("data/models/expr")
VOCAB_PATH = EXPR_ROOT / "vocab.json"
DEFAULT_DATASETS = ("cosmx_nsclc", "merfish_hypothalamus", "starmap")


def gene_key(name: str) -> str:
    return str(name).upper()


def build_vocab(datasets=DEFAULT_DATASETS, path: Path = VOCAB_PATH) -> list[str]:
    from spatial_reason.data.datasets import list_samples, load_sample

    genes = set()
    for ds in datasets:
        for s in list_samples(ds):
            genes |= {gene_key(g) for g in load_sample(ds, s).var_names}
    vocab = sorted(genes)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"datasets": list(datasets), "genes": vocab}, indent=1))
    print(f"[expr] vocab {len(vocab)} genes from {list(datasets)} -> {path}")
    return vocab


def load_vocab(path: Path | str = VOCAB_PATH) -> list[str]:
    path = Path(path)
    if not path.exists():
        return build_vocab(path=path)
    return json.loads(path.read_text())["genes"]


@register("expr")
class ExpressionEncoder(SpatialEncoder):
    dtype = np.float16

    def __init__(self, clip: tuple[float, float] = (-5.0, 10.0), min_std: float = 0.05,
                 split_config: str = DEFAULT_SPLIT_CONFIG):
        self.clip, self.min_std, self.split_config = clip, min_std, split_config
        self.vocab = load_vocab()
        self.col = {g: i for i, g in enumerate(self.vocab)}
        self._stats: dict[str, dict] = {}

    def stats(self, dataset: str) -> dict:
        if dataset in self._stats:
            return self._stats[dataset]
        path = EXPR_ROOT / f"stats_{dataset}.npz"
        if not path.exists():
            from spatial_reason.data.datasets import load_sample

            samples = train_samples(dataset, self.split_config)
            n, s1, s2, genes = 0, 0.0, 0.0, None
            for s in samples:
                a = load_sample(dataset, s)
                g = [gene_key(x) for x in a.var_names]
                assert genes is None or g == genes, f"{dataset}: 各切片基因顺序不一致"
                genes = g
                L = log_expr(a).astype(np.float64)
                n, s1, s2 = n + len(L), s1 + L.sum(0), s2 + (L ** 2).sum(0)
            mean = s1 / n
            std = np.sqrt(np.maximum(s2 / n - mean ** 2, 0))
            np.savez(path, genes=np.array(genes), mean=mean, std=std, train_samples=np.array(samples))
            print(f"[expr] {dataset}: stats from {samples} ({n} cells) -> {path}")
        d = np.load(path)
        self._stats[dataset] = {k: d[k] for k in d.files}
        return self._stats[dataset]

    def encode(self, adata: ad.AnnData, dataset: str) -> tuple[np.ndarray, dict]:
        st = self.stats(dataset)
        genes = [gene_key(x) for x in adata.var_names]
        if genes != st["genes"].tolist():
            raise ValueError(f"{dataset}: 基因顺序与训练切片统计量不一致")
        z = (log_expr(adata) - st["mean"]) / np.maximum(st["std"], self.min_std)
        z = np.clip(z, *self.clip)
        out = np.zeros((adata.n_obs, len(self.vocab)), dtype=self.dtype)
        out[:, [self.col[g] for g in genes]] = z
        return out, {"vocab": str(VOCAB_PATH), "n_vocab": len(self.vocab), "n_measured": len(genes),
                     "clip": list(self.clip), "train_samples": st["train_samples"].tolist()}
