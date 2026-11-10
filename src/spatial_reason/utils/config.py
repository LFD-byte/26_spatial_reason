"""读取并合并 configs/ 下的 YAML。"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml


def load_yaml(path: Path | str) -> dict[str, Any]:
    with open(path) as f:
        return yaml.safe_load(f) or {}


def deep_merge(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def load_config(*paths: Path | str, overrides: list[str] | None = None) -> dict[str, Any]:
    """按顺序合并多个 YAML;overrides 形如 ["train.lr=1e-4", "data.k_neighbors=64"]。"""
    cfg: dict[str, Any] = {}
    for p in paths:
        cfg = deep_merge(cfg, load_yaml(p))
    for item in overrides or []:
        key, val = item.split("=", 1)
        node = cfg
        *parents, leaf = key.split(".")
        for p in parents:
            node = node.setdefault(p, {})
        node[leaf] = yaml.safe_load(val)
    return cfg
