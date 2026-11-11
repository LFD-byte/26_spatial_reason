"""阶段 1 第 1 步:构造教师请求(训练环境)。

    python scripts/05_build_teacher_prompts.py --config configs/train/stage1_sft.yaml
输出:{traces.dir}/requests_{train,val}.jsonl 和 panel_genes_{dataset}.json
"""

import argparse
from pathlib import Path

from spatial_reason.traces.generate import build_requests
from spatial_reason.utils.config import load_config


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/train/stage1_sft.yaml")
    p.add_argument("--override", nargs="*", default=[])
    args = p.parse_args()
    cfg = load_config(args.config, overrides=args.override)
    out = Path(cfg["traces"]["dir"])
    for split in ("train", "val"):
        n = build_requests(cfg, split, out / f"requests_{split}.jsonl")
        print(f"[ok] {split}: {n} requests -> {out / f'requests_{split}.jsonl'}")


if __name__ == "__main__":
    main()
