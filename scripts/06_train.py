"""训练入口:模型配置 + 阶段配置合并后按 stage 分派。

    # 双卡(HF Accelerate + DDP)
    accelerate launch --config_file configs/accelerate/ddp_2gpu.yaml scripts/06_train.py \
        --config configs/train/stage0_align.yaml
    # 单卡
    python scripts/06_train.py --config configs/train/stage0_align.yaml \
        --override train.max_steps=50 train.eval_every=25 train.output_dir=outputs/stage0/smoke
    # 只训练 DLPFC:叠加 dlpfc.yaml 并筛选数据集
    python scripts/06_train.py --config configs/train/stage1_sft.yaml configs/train/dlpfc.yaml --datasets dlpfc
"""

import argparse

from spatial_reason.utils.config import load_config


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", nargs="+", required=True, help="多个文件按顺序合并")
    p.add_argument("--datasets", nargs="*", default=None, help="只用这些数据集(默认配置里的全部)")
    p.add_argument("--model-config", default="configs/model/qwen3_4b.yaml")
    p.add_argument("--override", nargs="*", default=[])
    args = p.parse_args()

    cfg = load_config(args.model_config, *args.config, overrides=args.override)
    if args.datasets:
        cfg["data"]["datasets"] = {k: v for k, v in cfg["data"]["datasets"].items() if k in args.datasets}
        assert cfg["data"]["datasets"], f"配置里没有 {args.datasets}"
    stage = cfg["stage"]
    if stage == 0:
        from spatial_reason.training.stage0_align import train
    elif stage == 1:
        from spatial_reason.training.stage1_sft import train
    else:
        raise NotImplementedError(f"stage {stage} 尚未实现")
    train(cfg)


if __name__ == "__main__":
    main()
