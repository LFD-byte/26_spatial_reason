"""阶段 1 第 1 步:构造教师请求(训练环境)。

    python scripts/05_build_teacher_prompts.py --config configs/train/stage1_sft.yaml
    # 只为部分数据集构造(多个配置按顺序合并),例如 DLPFC:
    python scripts/05_build_teacher_prompts.py --config configs/train/stage1_sft.yaml configs/train/dlpfc.yaml \
        --datasets dlpfc --override data.encoder=pca traces.dir=data/traces/qwen3_32b_dlpfc
输出:{traces.dir}/requests_{train,val}.jsonl 和 panel_genes_{dataset}.json
"""

import argparse
from pathlib import Path

from spatial_reason.traces.generate import build_requests
from spatial_reason.utils.config import load_config


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", nargs="+", default=["configs/train/stage1_sft.yaml"])
    p.add_argument("--override", nargs="*", default=[])
    p.add_argument("--datasets", nargs="*", default=None, help="只构造这些数据集(默认配置里的全部)")
    args = p.parse_args()
    cfg = load_config(*args.config, overrides=args.override)
    if args.datasets:
        cfg["data"]["datasets"] = {k: v for k, v in cfg["data"]["datasets"].items() if k in args.datasets}
        assert cfg["data"]["datasets"], f"配置里没有 {args.datasets}"
    out = Path(cfg["traces"]["dir"])
    for split in ("train", "val"):
        n = build_requests(cfg, split, out / f"requests_{split}.jsonl")
        print(f"[ok] {split}: {n} requests -> {out / f'requests_{split}.jsonl'}")


if __name__ == "__main__":
    main()
