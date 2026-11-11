"""在完整验证集上评测 stage 1 的 checkpoint(训练中只评测分层子集,挑 best 时样本少、方差大)。

    accelerate launch --config_file configs/accelerate/ddp_2gpu.yaml scripts/07_eval_stage1.py \
        --run-dir outputs/stage1/novae_qwen3_32b_teacher --tags best last

输出 {run_dir}/eval_full_{tag}.json(真实嵌入 + 打乱嵌入对照)与 predictions_full_{tag}[_shuffled].jsonl。
"""

import argparse
import json
from pathlib import Path

from accelerate import Accelerator

from spatial_reason.models.spatial_llm import SpatialLLM
from spatial_reason.training.stage1_sft import Stage1Dataset, evaluate_classification, load_records


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--run-dir", required=True)
    p.add_argument("--tags", nargs="+", default=["best", "last"])
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--no-shuffled", action="store_true")
    args = p.parse_args()

    run = Path(args.run_dir)
    cfg = json.loads((run / "config.json").read_text())
    mcfg, tcfg = cfg["model"], cfg["train"]
    acc = Accelerator()
    _, _, val_all, slices = load_records(cfg)
    ds = Stage1Dataset(val_all, slices, cfg["data"]["k_neighbors"])
    embed_dim = next(iter(slices.values())).emb.shape[1]
    model = SpatialLLM(mcfg["llm"], embed_dim=embed_dim, use_position=mcfg.get("use_position", True),
                       gradient_checkpointing=False, device=acc.device, lora=mcfg["lora"])
    kw = dict(n=10 ** 9, batch_size=args.batch_size, max_new_tokens=tcfg.get("max_new_tokens", 384), accelerator=acc)

    for tag in args.tags:
        ck = model.load_projector(run / tag / "projector.pt")
        model.load_lora(run / tag / "lora")
        res = {"tag": tag, "step": ck.get("step"), "n": len(ds)}
        for shuffled in ([False] if args.no_shuffled else [False, True]):
            summary, rows = evaluate_classification(model, ds, shuffled=shuffled, **kw)
            res.update(summary)
            if acc.is_main_process:
                name = f"predictions_full_{tag}{'_shuffled' if shuffled else ''}.jsonl"
                with open(run / name, "w") as f:
                    f.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)
        if acc.is_main_process:
            (run / f"eval_full_{tag}.json").write_text(json.dumps(res, indent=2))
            print(f"[{tag} @ step {res['step']}] " + "  ".join(
                f"{k}={v:.3f}" for k, v in res.items() if k.endswith("/acc") or k.endswith("/macro_f1")))


if __name__ == "__main__":
    main()
