"""在每个数据集的训练切片上自监督微调 Novae(或 --from-scratch 从头训练)(SwAV,不用 niche 标签),保存到 data/models/novae_ft/{dataset}。

只用训练切片:验证切片既不参与微调也不参与原型初始化,避免任何形式的泄漏。之后用
`scripts/02_extract_embeddings.py --encoder novae_ft` 在全部切片上提取嵌入。

    CUDA_VISIBLE_DEVICES=0 python scripts/02b_finetune_novae.py --datasets cosmx_nsclc
    CUDA_VISIBLE_DEVICES=1 python scripts/02b_finetune_novae.py --datasets merfish_hypothalamus starmap
"""

import argparse
import json
import time
from pathlib import Path

from spatial_reason.data.datasets import list_samples, load_sample
from spatial_reason.encoders.novae import DATASET_SETTINGS, FT_ROOT, SCRATCH_ROOT, prepare_for_novae
from spatial_reason.utils.config import load_config


def train_samples(cfg: dict, dataset: str) -> list[str]:
    val = set(cfg["data"]["datasets"][dataset]["val_samples"])
    return [s for s in list_samples(dataset) if s not in val]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--datasets", nargs="+", required=True)
    p.add_argument("--config", default="configs/train/stage1_sft.yaml", help="取 val_samples,与训练划分一致")
    p.add_argument("--max-epochs", type=int, default=20)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--min-prototypes-ratio", type=float, default=0.3)
    # Novae 默认早停 min_delta=0.05、patience=4,对小数据集太粗:MERFISH / STARmap 只跑 5–6 个 epoch 就停了
    p.add_argument("--min-delta", type=float, default=0.005)
    p.add_argument("--patience", type=int, default=8)
    p.add_argument("--num-workers", type=int, default=8, help="dataloader 进程数;CosMx 69 万细胞单进程建子图极慢")
    p.add_argument("--from-scratch", action="store_true",
                   help="不加载预训练权重,只在本数据集训练切片上从头自监督训练(输出到 data/models/novae_scratch)")
    p.add_argument("--tag", default="", help="输出目录后缀,便于比较不同微调设置")
    args = p.parse_args()
    cfg = load_config(args.config)

    import novae

    for ds in args.datasets:
        s = DATASET_SETTINGS[ds]
        samples = train_samples(cfg, ds)
        adatas = [prepare_for_novae(load_sample(ds, x), ds, s, slide_key=x)[0] for x in samples]
        print(f"[ft] {ds}: base {s['model']}, train sections {samples} "
              f"({sum(a.n_obs for a in adatas)} cells)", flush=True)
        t0 = time.time()
        common = dict(max_epochs=args.max_epochs, accelerator="cuda", lr=args.lr, min_delta=args.min_delta,
                      patience=args.patience, num_workers=args.num_workers, devices=1)
        if args.from_scratch:  # 默认架构;基因嵌入由本数据集表达的 PCA 初始化
            model = novae.Novae(adatas, min_prototypes_ratio=args.min_prototypes_ratio)
            model.fit(**common)
            root = SCRATCH_ROOT
        else:
            model = novae.Novae.from_pretrained(f"data/models/{s['model']}")
            model.fine_tune(adatas, min_prototypes_ratio=args.min_prototypes_ratio, **common)
            root = FT_ROOT
        out = Path(root) / (ds + args.tag)
        model.save_pretrained(out)
        (out / "finetune_info.json").write_text(json.dumps(
            {"base": None if args.from_scratch else s["model"], "from_scratch": args.from_scratch,
             "train_samples": samples, "max_epochs": args.max_epochs, "lr": args.lr,
             "min_prototypes_ratio": args.min_prototypes_ratio, "min_delta": args.min_delta,
             "patience": args.patience, "minutes": round((time.time() - t0) / 60, 1)},
            indent=2))
        print(f"[ft] {ds} done in {(time.time() - t0) / 60:.1f} min -> {out}", flush=True)


if __name__ == "__main__":
    main()
