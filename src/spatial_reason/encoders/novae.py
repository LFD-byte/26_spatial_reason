"""Novae 封装:零样本(novae)或在训练切片上自监督微调后(novae_ft)提取每个细胞的 64 维表示。

每个数据集的设置:
- 模型:人类组织用 novae-human-0,小鼠用 novae-mouse-0;
- 半径与坐标同单位。CosMx / MERFISH 为 µm,STARmap / Visium 为像素,用"中位最近邻距离 × 4"自动确定;
- MERFISH 的 X 非整数(按细胞体积归一化),Novae 会误认为已 log 化,需先 normalize_total + log1p。
"""

from __future__ import annotations

import anndata as ad
import numpy as np

from spatial_reason.encoders.base import SpatialEncoder, register

DATASET_SETTINGS = {
    "cosmx_nsclc": {"model": "novae-human-0", "radius": 80.0, "log1p": False},
    "merfish_hypothalamus": {"model": "novae-mouse-0", "radius": 100.0, "log1p": True},
    "starmap": {"model": "novae-mouse-0", "radius": "auto", "log1p": False},
    "dlpfc": {"model": "novae-human-0", "radius": "auto", "log1p": False},
}


def auto_radius(xy: np.ndarray, factor: float = 4.0) -> float:
    from sklearn.neighbors import NearestNeighbors

    d = NearestNeighbors(n_neighbors=2).fit(xy).kneighbors()[0][:, 1]
    return float(np.median(d) * factor)


@register("novae")
class NovaeEncoder(SpatialEncoder):
    def __init__(self, model_root: str = "data/models", accelerator: str = "cuda",
                 settings: dict | None = None):
        self.model_root, self.accelerator = model_root, accelerator
        self.settings = settings or DATASET_SETTINGS
        self._models: dict = {}

    def _model(self, name: str):
        import novae

        if name not in self._models:
            self._models[name] = novae.Novae.from_pretrained(f"{self.model_root}/{name}")
        return self._models[name]

    def encode(self, adata: ad.AnnData, dataset: str) -> tuple[np.ndarray, dict]:
        s = self.settings[dataset]
        a, radius = prepare_for_novae(adata, dataset, s)
        self._model(s["model"]).compute_representations(a, zero_shot=True, accelerator=self.accelerator)
        z = np.asarray(a.obsm["novae_latent"], dtype=np.float32)
        return z, {"model": s["model"], "radius": radius, "log1p": s["log1p"]}


def prepare_for_novae(adata: ad.AnnData, dataset: str, s: dict | None = None,
                      slide_key: str | None = None) -> tuple[ad.AnnData, float]:
    """按数据集设置做 Novae 所需预处理(可选 log1p)并建空间图。零样本与微调共用,保证只差在权重上。"""
    import novae
    import scanpy as sc

    s = s or DATASET_SETTINGS[dataset]
    a = ad.AnnData(X=adata.X.copy(), obs=adata.obs[[]].copy(), var=adata.var[[]].copy())
    a.obsm["spatial"] = adata.obsm["spatial"]
    if slide_key:
        a.obs["slide_id"] = slide_key
    if s["log1p"]:
        sc.pp.normalize_total(a)
        sc.pp.log1p(a)
    radius = auto_radius(a.obsm["spatial"]) if s["radius"] == "auto" else float(s["radius"])
    novae.spatial_neighbors(a, radius=radius)
    return a, radius


FT_ROOT = "data/models/novae_ft"
SCRATCH_ROOT = "data/models/novae_scratch"


@register("novae_ft")
class NovaeFineTunedEncoder(NovaeEncoder):
    """在本数据集训练切片上自监督微调过的 Novae(scripts/02b_finetune_novae.py);不用 niche 标签。"""

    root = FT_ROOT

    def _model(self, name: str):
        import novae

        if name not in self._models:
            self._models[name] = novae.Novae.from_pretrained(f"{self.root}/{name}")
        return self._models[name]

    def encode(self, adata: ad.AnnData, dataset: str) -> tuple[np.ndarray, dict]:
        s = self.settings[dataset]
        a, radius = prepare_for_novae(adata, dataset, s)
        self._model(dataset).compute_representations(a, zero_shot=False, accelerator=self.accelerator)
        z = np.asarray(a.obsm["novae_latent"], dtype=np.float32)
        return z, {"model": f"{self.root}/{dataset}", "base": s["model"], "radius": radius, "log1p": s["log1p"]}


@register("novae_scratch")
class NovaeScratchEncoder(NovaeFineTunedEncoder):
    """不用预训练权重、只在本数据集训练切片上从头训练的 Novae(02b_finetune_novae.py --from-scratch)。"""

    root = SCRATCH_ROOT
