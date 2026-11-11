"""阶段 0:冻结 LLM,只训练投影层(看嵌入写描述)。

评测时除了 val loss,还算"打乱对照":把 batch 内各样本的细胞嵌入整体错位,
文字 prompt 不变。如果打乱后 loss 不升高,说明 LLM 没有在用嵌入,只是在背描述模板 / 文本先验。
另外对生成的描述做解析,与真值比较细胞类型、占比和富集基因。
"""

from __future__ import annotations

import json
import math
import re
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler

from spatial_reason.data.collate import (Collator, Stage0Dataset, dataset_balanced_weights, load_slice,
                                         sample_centers)
from spatial_reason.data.splits import split_samples
from spatial_reason.models.spatial_llm import SpatialLLM
from spatial_reason.utils.logging import JsonlLogger
from spatial_reason.utils.seed import set_seed


# ---------------------------------------------------------------- 数据

def build_datasets(cfg: dict) -> tuple[Stage0Dataset, Stage0Dataset]:
    dcfg = cfg["data"]
    train_slices, val_slices, train_max, val_max = [], [], [], []
    for ds, opts in dcfg["datasets"].items():
        split = split_samples(ds, opts.get("val_samples", []))
        for s in split["train"]:
            train_slices.append(load_slice(ds, s, dcfg["encoder"], dcfg["k_neighbors"]))
            train_max.append(opts.get("max_centers_per_sample"))
        for s in split["val"]:
            val_slices.append(load_slice(ds, s, dcfg["encoder"], dcfg["k_neighbors"]))
            val_max.append(dcfg.get("val_centers_per_sample", 200))
        print(f"[data] {ds}: train {split['train']} | val {split['val']}")

    seed = cfg["train"]["seed"]
    train_centers, val_centers = [], []
    for i, (s, m) in enumerate(zip(train_slices, train_max)):
        train_centers += [(i, c) for _, c in sample_centers([s], m, seed + i)]
    for i, (s, m) in enumerate(zip(val_slices, val_max)):
        val_centers += [(i, c) for _, c in sample_centers([s], m, seed + 1000 + i)]
    print(f"[data] train centers {len(train_centers)}, val centers {len(val_centers)}")
    return (Stage0Dataset(train_slices, train_centers, train=True, seed=seed),
            Stage0Dataset(val_slices, val_centers, train=False, seed=seed))


# ---------------------------------------------------------------- 评分

_PCT = re.compile(r"(\d{1,3})%\s+([A-Za-z0-9+/\- ]+?)(?=,| and |\.|;|$)")


def score_description(text: str, meta: dict, gene_vocab: set[str]) -> dict:
    """解析生成的描述,与真值比较。只对有细胞类型的数据集计算组成相关指标。"""
    out = {}
    true_genes = set(meta["genes"])
    pred_genes = {w.strip(",.;:") for w in text.split()} & gene_vocab
    if pred_genes or true_genes:
        inter = len(pred_genes & true_genes)
        out["gene_precision"] = inter / len(pred_genes) if pred_genes else 0.0
        out["gene_recall"] = inter / len(true_genes) if true_genes else 1.0
    if meta["cell_types"]:
        truth = dict(meta["cell_types"])
        pred = {}
        for pct, name in _PCT.findall(text):
            pred.setdefault(name.strip(), int(pct))
        tp = len(set(pred) & set(truth))
        prec = tp / len(pred) if pred else 0.0
        rec = tp / len(truth)
        out["type_f1"] = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        top_true = max(truth, key=truth.get)
        out["top_type_acc"] = float(bool(pred) and max(pred, key=pred.get) == top_true)
        common = set(pred) & set(truth)
        if common:
            out["pct_mae"] = float(np.mean([abs(pred[n] - truth[n]) for n in common]))
        if meta.get("center_type"):
            c = meta["center_type"]
            out["center_acc"] = float(c in text or re.sub(r"s$", "", c) in text)
    return out


def _move(batch: dict, device) -> dict:
    return {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}


def _shuffle_cells(batch: dict) -> dict:
    """把细胞嵌入在 batch 内整体错位一位(每个样本拿到别人的邻域);坐标同步错位。"""
    b = dict(batch)
    b["cell_emb"] = torch.roll(batch["cell_emb"], 1, 0)
    b["rel_pos"] = torch.roll(batch["rel_pos"], 1, 0)
    return b


@torch.no_grad()
def evaluate_loss(model: SpatialLLM, dataset: Stage0Dataset, batch_size: int, accelerator=None,
                  max_batches: int | None = None) -> dict:
    """val 样本按进程切分,各自累加 token 加权的 loss,再跨进程求和。"""
    model.eval()
    rank, world = (accelerator.process_index, accelerator.num_processes) if accelerator else (0, 1)
    shard = torch.utils.data.Subset(dataset, list(range(rank, len(dataset), world)))
    loader = DataLoader(shard, batch_size=batch_size, shuffle=False, collate_fn=Collator(model.tokenizer))
    tot = torch.zeros(3, device=model.device)  # normal, shuffled, n_tok
    for bi, batch in enumerate(loader):
        if max_batches and bi >= max_batches:
            break
        batch = _move(batch, model.device)
        ntok = (batch["labels"] != -100).sum()
        for j, b in enumerate([batch, _shuffle_cells(batch)]):
            out = model(b["input_ids"], b["attention_mask"], b["cell_emb"], b["rel_pos"], labels=b["labels"])
            tot[j] += out.loss.float() * ntok
        tot[2] += ntok
    if accelerator:
        tot = accelerator.reduce(tot, reduction="sum")
    model.train()
    return {"val_loss": (tot[0] / tot[2]).item(), "val_loss_shuffled": (tot[1] / tot[2]).item()}


@torch.no_grad()
def evaluate_generation(model: SpatialLLM, dataset: Stage0Dataset, n: int, batch_size: int,
                        max_new_tokens: int = 96, shuffled: bool = False,
                        accelerator=None) -> tuple[dict, list[dict]]:
    """生成样本按进程切分,结果用 gather_object 收集到所有进程。"""
    model.eval()
    gene_vocab = set()
    for s in dataset.slices:
        gene_vocab |= set(map(str, s.desc.genes))
    collate = Collator(model.tokenizer, with_target=False)
    idx = np.linspace(0, len(dataset) - 1, min(n, len(dataset))).astype(int)
    if accelerator:
        idx = idx[accelerator.process_index::accelerator.num_processes]
    scores, examples = [], []
    for s in range(0, len(idx), batch_size):
        batch = _move(collate([dataset[int(i)] for i in idx[s:s + batch_size]]), model.device)
        if shuffled:
            batch = _shuffle_cells(batch)
        gen = model.generate(batch["input_ids"], batch["attention_mask"], batch["cell_emb"], batch["rel_pos"],
                             max_new_tokens=max_new_tokens, do_sample=False)
        texts = model.tokenizer.batch_decode(gen, skip_special_tokens=True)
        for t, m, ref in zip(texts, batch["meta"], batch["target"]):
            sc = score_description(t, m, gene_vocab)
            scores.append(sc)
            examples.append({"dataset": m["dataset"], "sample": m["sample"], "center": m["center"],
                             "reference": ref, "generated": t.strip(), **sc})
    model.train()
    if accelerator:
        from accelerate.utils import gather_object

        examples = sorted(gather_object(examples), key=lambda e: (e["dataset"], e["sample"], e["center"]))
        scores = [{k: e[k] for k in e if k not in ("dataset", "sample", "center", "reference", "generated")}
                  for e in examples]
    keys = sorted({k for s in scores for k in s})
    prefix = "gen_shuf_" if shuffled else "gen_"
    summary = {prefix + k: float(np.mean([s[k] for s in scores if k in s])) for k in keys}
    return summary, examples


# ---------------------------------------------------------------- 训练

def lr_lambda(step: int, warmup: int, total: int) -> float:
    if step < warmup:
        return (step + 1) / warmup
    progress = (step - warmup) / max(1, total - warmup)
    return 0.5 * (1 + math.cos(math.pi * min(1.0, progress)))


def _report_to(value) -> list[str]:
    """train.report_to: "none" | "wandb" | "tensorboard" | 列表。"""
    items = [value] if isinstance(value, str) else list(value or [])
    items = [v for v in items if v and v != "none"]
    unknown = set(items) - {"wandb", "tensorboard"}
    if unknown:
        raise ValueError(f"不支持的 report_to: {sorted(unknown)}")
    return items


def _flatten(d: dict, prefix: str = "") -> dict:
    """嵌套配置展平成 a.b.c 形式,便于在 wandb / TensorBoard 里筛选对比。"""
    out = {}
    for k, v in d.items():
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            out.update(_flatten(v, key + "."))
        else:
            out[key] = v if isinstance(v, (int, float, str, bool)) or v is None else str(v)
    return out


def train(cfg: dict) -> Path:
    """单卡 / 多卡通用。多卡用 `accelerate launch`,后端由 configs/accelerate/*.yaml 决定(阶段 0 用 DDP)。

    batch:train.batch_size 是每张卡的 micro batch;train.global_batch_size 是每个优化器步的总样本数,
    梯度累积步数 = global_batch_size / (batch_size × 卡数)。
    """
    from accelerate import Accelerator
    from accelerate.utils import set_seed as acc_set_seed

    import os

    tcfg, mcfg = cfg["train"], cfg["model"]
    world = int(os.environ.get("WORLD_SIZE", "1"))
    gbs = tcfg.get("global_batch_size", tcfg["batch_size"] * tcfg.get("grad_accum", 1))
    if gbs % (tcfg["batch_size"] * world):
        raise ValueError(f"global_batch_size {gbs} 不能被 batch_size × 卡数 {tcfg['batch_size'] * world} 整除")
    accum = gbs // (tcfg["batch_size"] * world)
    report_to = _report_to(tcfg.get("report_to", "none"))
    out_dir = Path(tcfg["output_dir"])
    accelerator = Accelerator(gradient_accumulation_steps=accum, log_with=report_to or None,
                              project_dir=str(out_dir))
    assert accelerator.num_processes == world
    acc_set_seed(tcfg["seed"])
    set_seed(tcfg["seed"])
    main = accelerator.is_main_process

    runtime = {"num_processes": world, "grad_accum": accum, "distributed_type": str(accelerator.distributed_type)}
    if main:
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "config.json").write_text(json.dumps({**cfg, "runtime": runtime}, indent=2, ensure_ascii=False))
    logger = JsonlLogger(out_dir / "metrics.jsonl") if main else None
    if report_to:
        # 只有主进程真正创建 tracker;wandb 的本地文件放在 output_dir 下
        accelerator.init_trackers(
            tcfg.get("project", "spatial_reason"),
            config=_flatten({**cfg, "runtime": runtime}),
            init_kwargs={"wandb": {"name": tcfg.get("run_name") or out_dir.name, "dir": str(out_dir),
                                   "tags": [f"stage{cfg['stage']}", cfg["data"]["encoder"]]}},
        )

    def log(split: str, step: int, **rec):
        if main:
            logger.log(split=split, step=step, **rec)
            if report_to:
                accelerator.log({f"{split}/{k}": v for k, v in rec.items() if isinstance(v, (int, float))},
                                step=step)

    def log_samples(step: int, examples: list[dict], name: str):
        """生成样例写成 wandb 表格(TensorBoard 不支持表格,只记 jsonl)。"""
        if not (main and "wandb" in report_to and examples):
            return
        import wandb

        cols = ["dataset", "sample", "center", "reference", "generated", "type_f1", "gene_recall"]
        table = wandb.Table(columns=cols, data=[[e.get(c) for c in cols] for e in examples[:100]])
        accelerator.get_tracker("wandb", unwrap=True).log({f"val/{name}": table}, step=step)

    train_ds, val_ds = build_datasets(cfg)
    embed_dim = train_ds.slices[0].emb.shape[1]
    model = SpatialLLM(mcfg["llm"], embed_dim=embed_dim, use_position=mcfg.get("use_position", True),
                       gradient_checkpointing=mcfg.get("gradient_checkpointing", False),
                       device=accelerator.device)
    if main:
        n_params = sum(p.numel() for p in model.projector.parameters())
        print(f"[model] projector params {n_params / 1e6:.2f}M, output scale {model.projector.scale.item():.4f}")
        print(f"[dist] {accelerator.distributed_type} x{world}, micro batch {tcfg['batch_size']}, "
              f"grad accum {accum}, global batch {gbs}")

    collate = Collator(model.tokenizer, max_length=tcfg.get("max_length", 512))
    nw = tcfg.get("num_workers", 4)
    alpha = cfg["data"].get("sampling_alpha", 1.0)
    if alpha == 1.0:
        sampler_kw = {"shuffle": True}
    else:
        # 各进程主种子相同 -> 采样序列相同,accelerate 再按 batch 切给各卡
        w = dataset_balanced_weights(train_ds, alpha)
        sampler_kw = {"sampler": WeightedRandomSampler(torch.as_tensor(w, dtype=torch.double),
                                                       num_samples=len(train_ds), replacement=True)}
        if main:
            names = np.array([train_ds.slices[si].dataset for si, _ in train_ds.centers])
            share = {d: round(float(w[names == d].sum() / w.sum()), 3) for d in np.unique(names)}
            print(f"[data] sampling_alpha={alpha}, dataset shares {share}")
    train_loader = DataLoader(train_ds, batch_size=tcfg["batch_size"], collate_fn=collate,
                              num_workers=nw, drop_last=True, persistent_workers=nw > 0, **sampler_kw)

    params = [p for p in model.projector.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=tcfg["lr"], weight_decay=tcfg.get("weight_decay", 0.0),
                            betas=(0.9, 0.95))
    total = tcfg["max_steps"]
    warmup = max(1, int(total * tcfg.get("warmup_ratio", 0.03)))
    # 调度器不交给 accelerate.prepare:它会按卡数多步进;这里每个优化器步手动 step 一次
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: lr_lambda(s, warmup, total))
    # DDP 只包装需要梯度的参数(投影层),冻结的 LLM 不参与梯度同步
    ddp_model, opt, train_loader = accelerator.prepare(model, opt, train_loader)
    raw = accelerator.unwrap_model(ddp_model)

    def run_eval(step: int, final: bool = False) -> float:
        m = evaluate_loss(raw, val_ds, tcfg["batch_size"], accelerator, tcfg.get("eval_max_batches"))
        n_gen = tcfg.get("n_gen_eval_final" if final else "n_gen_eval", 32)
        gbs_gen = tcfg.get("gen_batch_size", 16)
        g, ex = evaluate_generation(raw, val_ds, n_gen, gbs_gen, accelerator=accelerator)
        rec = {**m, "val_loss_gap": m["val_loss_shuffled"] - m["val_loss"], **g}
        if final:
            gs, ex_s = evaluate_generation(raw, val_ds, n_gen, gbs_gen, shuffled=True, accelerator=accelerator)
            rec.update(gs)
            log_samples(step, ex_s, "samples_shuffled")
            if main:
                with open(out_dir / "samples_final_shuffled.jsonl", "w") as f:
                    f.writelines(json.dumps(e, ensure_ascii=False) + "\n" for e in ex_s)
        log("val", step, **rec)
        log_samples(step, ex, "samples")
        if main:
            with open(out_dir / f"samples_step{step}.jsonl", "w") as f:
                f.writelines(json.dumps(e, ensure_ascii=False) + "\n" for e in ex)
            for e in ex[:2]:
                print(f"  [{e['dataset']}] REF: {e['reference']}\n  {' ' * len(e['dataset'])}   GEN: {e['generated']}")
        return m["val_loss"]

    def save(name: str, step: int):
        accelerator.wait_for_everyone()
        if main:
            raw.save_projector(out_dir / name, {"step": step, "config": cfg})

    best = run_eval(0)
    step, run_loss, n_micro = 0, torch.zeros((), device=accelerator.device), 0
    ddp_model.train()
    while step < total:
        train_ds.epoch += 1
        for batch in train_loader:
            batch = _move(batch, accelerator.device)
            with accelerator.accumulate(ddp_model):
                out = ddp_model(batch["input_ids"], batch["attention_mask"], batch["cell_emb"],
                                batch["rel_pos"], labels=batch["labels"])
                accelerator.backward(out.loss)
                run_loss += out.loss.detach()
                n_micro += 1
                if accelerator.sync_gradients:
                    gnorm = accelerator.clip_grad_norm_(params, tcfg.get("max_grad_norm", 1.0))
                opt.step()
                opt.zero_grad(set_to_none=True)
            if not accelerator.sync_gradients:
                continue
            sched.step()
            step += 1
            if step % tcfg.get("log_every", 10) == 0:
                loss = accelerator.reduce(run_loss / n_micro, reduction="mean").item()
                log(split="train", step=step, loss=loss, lr=sched.get_last_lr()[0],
                    grad_norm=float(gnorm), scale=raw.projector.scale.item())
                run_loss.zero_()
                n_micro = 0
            if step % tcfg["eval_every"] == 0 or step == total:
                vl = run_eval(step, final=step == total)
                save("projector_last.pt", step)
                if vl < best:
                    best = vl
                    save("projector_best.pt", step)
            if step >= total:
                break
    accelerator.wait_for_everyone()
    if main:
        print(f"[done] best val_loss {best:.4f} -> {out_dir}")
    accelerator.end_training()
    return out_dir
