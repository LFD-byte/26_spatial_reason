"""各数据集的统一加载接口(DLPFC Visium、MERFISH 下丘脑、STARmap、CosMx),输出 AnnData。"""

from __future__ import annotations

from pathlib import Path

import anndata as ad

PROCESSED_ROOT = Path("data/processed")

# 数据集元信息:prompt 中的组织描述、物种、平台
DATASET_INFO = {
    "cosmx_nsclc": {"tissue": "human non-small cell lung cancer", "species": "human", "platform": "CosMx"},
    "merfish_hypothalamus": {"tissue": "mouse hypothalamic preoptic region", "species": "mouse",
                             "platform": "MERFISH"},
    "starmap": {"tissue": "mouse visual cortex", "species": "mouse", "platform": "STARmap"},
    "dlpfc": {"tissue": "human dorsolateral prefrontal cortex", "species": "human", "platform": "Visium"},
}


def list_samples(dataset: str, root: Path | str = PROCESSED_ROOT) -> list[str]:
    return sorted(p.stem for p in (Path(root) / dataset).glob("*.h5ad"))


def load_sample(dataset: str, sample: str, root: Path | str = PROCESSED_ROOT) -> ad.AnnData:
    return ad.read_h5ad(Path(root) / dataset / f"{sample}.h5ad")
