"""SpatialEncoder 抽象基类与注册表;嵌入缓存到 data/embeddings/{encoder}/{dataset}/{sample}.npy。

缓存的嵌入行顺序与 data/processed/{dataset}/{sample}.h5ad 的 obs 顺序一致,
旁边的 {sample}.json 记录编码器参数与 obs_names 校验值。
"""

from __future__ import annotations

import hashlib
import json
from abc import ABC, abstractmethod
from pathlib import Path

import anndata as ad
import numpy as np

EMBED_ROOT = Path("data/embeddings")
_REGISTRY: dict[str, type["SpatialEncoder"]] = {}


def register(name: str):
    def deco(cls):
        _REGISTRY[name] = cls
        cls.name = name
        return cls
    return deco


def get_encoder(name: str, **kwargs) -> "SpatialEncoder":
    return _REGISTRY[name](**kwargs)


def obs_checksum(adata: ad.AnnData) -> str:
    return hashlib.md5("\n".join(map(str, adata.obs_names)).encode()).hexdigest()


class SpatialEncoder(ABC):
    name: str = "base"
    dtype = np.float32  # 缓存精度;表达输入(expr)用 float16

    @abstractmethod
    def encode(self, adata: ad.AnnData, dataset: str) -> tuple[np.ndarray, dict]:
        """返回 (n_obs, dim) float32 嵌入和参数记录。"""

    def cache_path(self, dataset: str, sample: str, root: Path = EMBED_ROOT) -> Path:
        return root / self.name / dataset / f"{sample}.npy"

    def encode_and_cache(self, adata: ad.AnnData, dataset: str, sample: str,
                         root: Path = EMBED_ROOT, overwrite: bool = False) -> Path:
        path = self.cache_path(dataset, sample, root)
        if path.exists() and not overwrite:
            print(f"[skip] {path}")
            return path
        z, info = self.encode(adata, dataset)
        assert z.shape[0] == adata.n_obs and np.isfinite(z).all(), "嵌入行数不符或含 NaN/inf"
        path.parent.mkdir(parents=True, exist_ok=True)
        np.save(path, z.astype(self.dtype))
        info = {"encoder": self.name, "dataset": dataset, "sample": sample, "shape": list(z.shape),
                "obs_checksum": obs_checksum(adata), **info}
        path.with_suffix(".json").write_text(json.dumps(info, indent=2))
        print(f"[ok] {path} {z.shape}")
        return path


def load_embeddings(encoder: str, dataset: str, sample: str, adata: ad.AnnData | None = None,
                    root: Path = EMBED_ROOT) -> np.ndarray:
    path = root / encoder / dataset / f"{sample}.npy"
    z = np.load(path)
    if adata is not None:
        info = json.loads(path.with_suffix(".json").read_text())
        if info["obs_checksum"] != obs_checksum(adata):
            raise ValueError(f"{path} 与 AnnData 的 obs 顺序不一致,请重新提取嵌入")
    return z
