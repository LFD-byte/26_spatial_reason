"""按切片 / 个体 / 平台划分 train/val/test(不按细胞随机划分)。"""

from __future__ import annotations

from spatial_reason.data.datasets import list_samples


def split_samples(dataset: str, val_samples: list[str], test_samples: list[str] | None = None,
                  ) -> dict[str, list[str]]:
    """按切片名划分。val/test 中的切片整张留出,其余都进 train。"""
    all_s = list_samples(dataset)
    test_samples = test_samples or []
    missing = set(val_samples + test_samples) - set(all_s)
    if missing:
        raise ValueError(f"{dataset} 中不存在切片: {sorted(missing)}")
    held = set(val_samples) | set(test_samples)
    return {"train": [s for s in all_s if s not in held], "val": list(val_samples), "test": list(test_samples)}
