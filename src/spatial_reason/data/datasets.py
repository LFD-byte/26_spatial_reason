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
    # 166 基因的 STARmap(BZ5/9/14)来自小鼠内侧前额叶皮层(mPFC),见 LLMiniST 配置
    "starmap": {"tissue": "mouse medial prefrontal cortex", "species": "mouse", "platform": "STARmap"},
    "dlpfc": {"tissue": "human dorsolateral prefrontal cortex", "species": "human", "platform": "Visium"},
}

# 空间域 / niche 标签:原始值 -> 给 LLM 看的名称。
# STARmap 的数字编码按 LLMiniST two_in_one_finetunePro_starmap.py 映射;MERFISH 缩写展开为全称。
DOMAIN_NAMES = {
    "starmap": {"1": "Layer 1", "2": "Layer 2/3", "3": "Layer 5", "4": "Layer 6"},
    "merfish_hypothalamus": {
        "BST": "BST (bed nucleus of the stria terminalis)",
        "MPA": "MPA (medial preoptic area)",
        "MPN": "MPN (medial preoptic nucleus)",
        "PV": "PV (periventricular hypothalamic nucleus)",
        "PVH": "PVH (paraventricular hypothalamic nucleus)",
        "PVT": "PVT (paraventricular nucleus of the thalamus)",
        "V3": "V3 (third ventricle)",
        "fx": "fx (fornix)",
    },
    "dlpfc": {f"Layer{i}": f"Layer {i}" for i in range(1, 7)} | {"WM": "white matter"},
}


def domain_name(dataset: str, raw) -> str | None:
    if raw is None or (isinstance(raw, float) and raw != raw):
        return None
    return DOMAIN_NAMES.get(dataset, {}).get(str(raw), str(raw))


def domain_candidates(dataset: str, raw_values) -> list[str]:
    """该数据集的候选标签(展示名),顺序固定。"""
    names = {domain_name(dataset, v) for v in raw_values}
    names.discard(None)
    return sorted(names)


def list_samples(dataset: str, root: Path | str = PROCESSED_ROOT) -> list[str]:
    return sorted(p.stem for p in (Path(root) / dataset).glob("*.h5ad"))


def load_sample(dataset: str, sample: str, root: Path | str = PROCESSED_ROOT) -> ad.AnnData:
    return ad.read_h5ad(Path(root) / dataset / f"{sample}.h5ad")
