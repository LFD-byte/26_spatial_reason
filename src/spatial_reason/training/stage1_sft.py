"""阶段 1:在过滤后的教师轨迹上做 SFT 冷启动(LoRA + 投影层)。

学生只看到细胞 token 和候选标签,学习教师基于文本数据写出的推理 / 证据 / 答案。
评测(按数据集分别汇总):
- 分类:在全部 val 邻域(不只是通过过滤的那些,避免偏向简单样本)上生成,算准确率、macro-F1;
- 格式合规率;
- 证据有效率:引用的细胞类型 / 基因中真正在该邻域富集的比例;
- 打乱对照:细胞 token 在 batch 内错位后的准确率。
"""

from __future__ import annotations

import json
import os
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from spatial_reason.data.collate import Collator, load_slice
from spatial_reason.data.datasets import DATASET_INFO
from spatial_reason.inference.generate import OUTPUT_FORMAT, match_label, parse_output
from spatial_reason.models.spatial_llm import CELL_TOKEN, SpatialLLM
from spatial_reason.traces.filter import _norm_type
from spatial_reason.training.checkpoint import is_done, load_checkpoint, mark_done, save_checkpoint
from spatial_reason.training.stage0_align import _flatten, _move, _report_to, _shuffle_cells, lr_lambda
from spatial_reason.utils.logging import JsonlLogger
from spatial_reason.utils.seed import set_seed

STUDENT_PROMPT = (
    "Below is a cell neighborhood from a {platform} section of {tissue}: <cells>{cells}</cells>\n"
    "The first cell token is the center cell and the other {k} are its nearest neighbors, ordered by distance.\n\n"
    "Candidate labels:\n{candidates}\n\n"
    "Which label does this neighborhood belong to? Reason from the cell types and genes you can read from "
    "the cell tokens, cite the enriched cell types and genes you relied on, and answer in exactly this format:\n"
    "{fmt}"
)


class Stage1Dataset(Dataset):
    """records:sft_*.jsonl(带 target)或 requests_*.jsonl(只评测,无 target)。"""

    def __init__(self, records: list[dict], slices: dict, k: int):
        self.records, self.slices, self.k = records, slices, k

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, i: int) -> dict:
        r = self.records[i]
        s = self.slices[(r["dataset"], r["sample"])]
        idx = s.nb.index[r["center"]]
        info = DATASET_INFO[r["dataset"]]
        prompt = STUDENT_PROMPT.format(platform=info["platform"], tissue=info["tissue"],
                                       cells=CELL_TOKEN * len(idx), k=len(idx) - 1,
                                       candidates="\n".join(f"- {c}" for c in r["candidates"]), fmt=OUTPUT_FORMAT)
        return {"cell_emb": torch.from_numpy(s.emb[idx]), "rel_pos": torch.from_numpy(s.nb.rel_pos[r["center"]]),
                "prompt": prompt, "target": r.get("target", ""), "meta": r}


def _read(path: Path) -> list[dict]:
    return [json.loads(line) for line in open(path)]


def load_records(cfg: dict) -> tuple[list[dict], list[dict], list[dict], dict]:
    d = Path(cfg["traces"]["dir"])
    train, val_sft, val_all = _read(d / "sft_train.jsonl"), _read(d / "sft_val.jsonl"), _read(d / "requests_val.jsonl")
    for r in val_all:
        r.pop("messages", None)
    need = {(r["dataset"], r["sample"]) for r in train + val_all}
    slices = {key: load_slice(key[0], key[1], cfg["data"]["encoder"], cfg["data"]["k_neighbors"], with_desc=False)
              for key in sorted(need)}
    return train, val_sft, val_all, slices


# ---------------------------------------------------------------- 评测

def score_prediction(text: str, rec: dict) -> dict:
    p = parse_output(text)
    pred = match_label(p.answer, rec["candidates"])
    st = rec["stats"]
    enriched_t = {_norm_type(t) for t in st["enriched_types"]}
    enriched_g = {g.upper() for g in st["enriched_genes"]}
    cited = [_norm_type(t) in enriched_t for t in p.evidence_types] + \
            [g.upper() in enriched_g for g in p.evidence_genes]
    return {"pred": pred, "correct": float(pred == rec["label"]), "well_formed": float(p.well_formed),
            "valid_answer": float(pred is not None),
            "evidence_valid": float(np.mean(cited)) if cited else float("nan"),
            "n_cited": len(cited)}


def macro_f1(y_true: list[str], y_pred: list[str | None]) -> float:
    labels = sorted(set(y_true))
    f1s = []
    for lab in labels:
        tp = sum(t == lab and p == lab for t, p in zip(y_true, y_pred))
        fp = sum(t != lab and p == lab for t, p in zip(y_true, y_pred))
        fn = sum(t == lab and p != lab for t, p in zip(y_true, y_pred))
        f1s.append(2 * tp / (2 * tp + fp + fn) if tp else 0.0)
    return float(np.mean(f1s))


@torch.no_grad()
def evaluate_classification(model: SpatialLLM, dataset: Stage1Dataset, n: int, batch_size: int,
                            max_new_tokens: int, accelerator=None, shuffled: bool = False) -> tuple[dict, list]:
    model.eval()
    collate = Collator(model.tokenizer, with_target=False)
    # 按数据集分层取样,保证每个数据集都被评测到
    by_ds = defaultdict(list)
    for i, r in enumerate(dataset.records):
        by_ds[r["dataset"]].append(i)
    per = max(1, n // len(by_ds))
    idx = sorted(i for ids in by_ds.values() for i in np.array(ids)[np.linspace(0, len(ids) - 1, min(per, len(ids))).astype(int)])
    if accelerator:
        idx = idx[accelerator.process_index::accelerator.num_processes]
    rows = []
    for s in range(0, len(idx), batch_size):
        items = [dataset[int(i)] for i in idx[s:s + batch_size]]
        batch = _move(collate(items), model.device)
        if shuffled:
            batch = _shuffle_cells(batch)
        gen = model.generate(batch["input_ids"], batch["attention_mask"], batch["cell_emb"], batch["rel_pos"],
                             max_new_tokens=max_new_tokens, do_sample=False)
        for t, it in zip(model.tokenizer.batch_decode(gen, skip_special_tokens=True), items):
            r = it["meta"]
            rows.append({"id": r["id"], "dataset": r["dataset"], "label": r["label"], "generated": t.strip(),
                         **score_prediction(t, r)})
    model.train()
    if accelerator:
        from accelerate.utils import gather_object

        rows = sorted(gather_object(rows), key=lambda x: x["id"])
    pre = "shuf_" if shuffled else ""
    summary = {}
    for ds in sorted({r["dataset"] for r in rows}) + ["all"]:
        rs = [r for r in rows if ds == "all" or r["dataset"] == ds]
        tag = f"{pre}{ds}"
        summary[f"{tag}/acc"] = float(np.mean([r["correct"] for r in rs]))
        if ds != "all":
            summary[f"{tag}/macro_f1"] = macro_f1([r["label"] for r in rs], [r["pred"] for r in rs])
        summary[f"{tag}/format"] = float(np.mean([r["well_formed"] for r in rs]))
        ev = [r["evidence_valid"] for r in rs if r["n_cited"]]
        summary[f"{tag}/evidence_valid"] = float(np.mean(ev)) if ev else 0.0
    return summary, rows


@torch.no_grad()
def evaluate_loss(model: SpatialLLM, dataset: Stage1Dataset, batch_size: int, max_length: int,
                  accelerator=None) -> dict:
    model.eval()
    rank, world = (accelerator.process_index, accelerator.num_processes) if accelerator else (0, 1)
    shard = torch.utils.data.Subset(dataset, list(range(rank, len(dataset), world)))
    loader = DataLoader(shard, batch_size=batch_size, collate_fn=Collator(model.tokenizer, max_length=max_length))
    tot = torch.zeros(3, device=model.device)
    for batch in loader:
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


# ---------------------------------------------------------------- 训练

def train(cfg: dict) -> Path:
    from accelerate import Accelerator
    from accelerate.utils import set_seed as acc_set_seed

    tcfg, mcfg = cfg["train"], cfg["model"]
    world = int(os.environ.get("WORLD_SIZE", "1"))
    gbs = tcfg["global_batch_size"]
    if gbs % (tcfg["batch_size"] * world):
        raise ValueError(f"global_batch_size {gbs} 不能被 batch_size × 卡数整除")
    accum = gbs // (tcfg["batch_size"] * world)
    report_to = _report_to(tcfg.get("report_to", "none"))
    out_dir = Path(tcfg["output_dir"])
    if is_done(out_dir):
        print(f"[skip] {out_dir} 已训练完成(存在 DONE);要重训请换 output_dir 或删除 DONE")
        return out_dir
    accelerator = Accelerator(gradient_accumulation_steps=accum, log_with=report_to or None, project_dir=str(out_dir))
    acc_set_seed(tcfg["seed"])
    set_seed(tcfg["seed"])
    main = accelerator.is_main_process
    runtime = {"num_processes": world, "grad_accum": accum}
    if main:
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "config.json").write_text(json.dumps({**cfg, "runtime": runtime}, indent=2, ensure_ascii=False))
    logger = JsonlLogger(out_dir / "metrics.jsonl") if main else None
    if report_to:
        accelerator.init_trackers(tcfg.get("project", "spatial_reason"), config=_flatten({**cfg, "runtime": runtime}),
                                  init_kwargs={"wandb": {"name": tcfg.get("run_name") or out_dir.name,
                                                         "dir": str(out_dir), "tags": ["stage1"]}})

    def log(split, step, **rec):
        if main:
            logger.log(split=split, step=step, **rec)
            if report_to:
                accelerator.log({f"{split}/{k}": v for k, v in rec.items() if isinstance(v, (int, float))}, step=step)

    train_recs, val_sft, val_all, slices = load_records(cfg)
    k = cfg["data"]["k_neighbors"]
    train_ds, val_loss_ds, val_cls_ds = (Stage1Dataset(r, slices, k) for r in (train_recs, val_sft, val_all))
    embed_dim = next(iter(slices.values())).emb.shape[1]
    model = SpatialLLM(mcfg["llm"], embed_dim=embed_dim, use_position=mcfg.get("use_position", True),
                       gradient_checkpointing=mcfg.get("gradient_checkpointing", True),
                       device=accelerator.device, lora=mcfg["lora"], gene_encoder=mcfg.get("gene_encoder"))
    if mcfg.get("init_projector"):
        ck = model.load_projector(mcfg["init_projector"])
        if main:
            print(f"[model] projector <- {mcfg['init_projector']} (stage 0 step {ck.get('step')})")
    if main:
        n_lora = sum(p.numel() for p in model.llm.parameters() if p.requires_grad)
        print(f"[model] trainable LoRA {n_lora / 1e6:.1f}M + projector "
              f"{sum(p.numel() for p in model.projector.parameters()) / 1e6:.1f}M")
        print(f"[data] train {len(train_ds)} traces, val loss {len(val_loss_ds)}, val classify {len(val_cls_ds)}")

    collate = Collator(model.tokenizer, max_length=tcfg.get("max_length", 1024))
    alpha = cfg["data"].get("sampling_alpha", 1.0)
    names = np.array([r["dataset"] for r in train_recs])
    uniq, counts = np.unique(names, return_counts=True)
    share = counts.astype(float) ** alpha / (counts.astype(float) ** alpha).sum()
    w = np.array([dict(zip(uniq, share / counts))[n] for n in names])
    steps_per_epoch = len(train_ds) // gbs
    total = steps_per_epoch * tcfg["epochs"]
    sampler = WeightedRandomSampler(torch.as_tensor(w, dtype=torch.double), num_samples=len(train_ds), replacement=True)
    nw = tcfg.get("num_workers", 4)
    train_loader = DataLoader(train_ds, batch_size=tcfg["batch_size"], sampler=sampler, collate_fn=collate,
                              num_workers=nw, drop_last=True, persistent_workers=nw > 0)
    if main:
        print(f"[data] dataset shares {dict(zip(uniq.tolist(), np.round(share, 3).tolist()))}; "
              f"{total} optimizer steps ({tcfg['epochs']} epochs)")

    lora_params = [p for p in model.llm.parameters() if p.requires_grad]
    proj_params = list(model.projector.parameters())
    opt = torch.optim.AdamW([{"params": lora_params, "lr": tcfg["lr"]},
                             {"params": proj_params, "lr": tcfg["projector_lr"]}],
                            weight_decay=tcfg.get("weight_decay", 0.0), betas=(0.9, 0.95))
    warmup = max(1, int(total * tcfg.get("warmup_ratio", 0.03)))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: lr_lambda(s, warmup, total))
    ddp_model, opt, train_loader = accelerator.prepare(model, opt, train_loader)
    raw = accelerator.unwrap_model(ddp_model)
    params = lora_params + proj_params

    def run_eval(step: int, final: bool = False) -> float:
        m = evaluate_loss(raw, val_loss_ds, tcfg["batch_size"], tcfg.get("max_length", 1024), accelerator)
        n = tcfg["n_gen_eval_final" if final else "n_gen_eval"]
        kw = dict(n=n, batch_size=tcfg.get("gen_batch_size", 16), max_new_tokens=tcfg.get("max_new_tokens", 384),
                  accelerator=accelerator)
        g, rows = evaluate_classification(raw, val_cls_ds, **kw)
        rec = {**m, "val_loss_gap": m["val_loss_shuffled"] - m["val_loss"], **g}
        if final:
            gs, rows_s = evaluate_classification(raw, val_cls_ds, shuffled=True, **kw)
            rec.update(gs)
            if main:
                with open(out_dir / "predictions_final_shuffled.jsonl", "w") as f:
                    f.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in rows_s)
        log("val", step, **rec)
        if main:
            with open(out_dir / f"predictions_step{step}.jsonl", "w") as f:
                f.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)
            print(f"  GEN [{rows[0]['dataset']} | {rows[0]['label']}]: {rows[0]['generated'][:600]}")
        return rec["all/acc"]

    def save(tag: str, step: int):
        accelerator.wait_for_everyone()
        if main:
            raw.save_projector(out_dir / tag / "projector.pt", {"step": step, "config": cfg})
            raw.save_lora(out_dir / tag / "lora")

    save_every = tcfg.get("save_every", 100)

    def checkpoint(step: int):
        save_checkpoint(out_dir, accelerator, raw, opt, sched, step, {"best": best})

    resumed = load_checkpoint(out_dir, raw, opt, sched, tcfg["seed"])
    if resumed:
        step, st = resumed
        best = st["best"]
        log("resume", step, best=best)
        if main:
            print(f"[resume] 从 step {step}/{total} 继续(best val acc {best:.4f})")
    else:
        run_eval(0)
        step, best = 0, -1.0  # 训练后第一次评测一定保存 best
    run_loss, n_micro = torch.zeros((), device=accelerator.device), 0
    ddp_model.train()
    while step < total:
        for batch in train_loader:
            batch = _move(batch, accelerator.device)
            with accelerator.accumulate(ddp_model):
                out = ddp_model(batch["input_ids"], batch["attention_mask"], batch["cell_emb"], batch["rel_pos"],
                                labels=batch["labels"])
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
                log("train", step, loss=loss, lr=sched.get_last_lr()[0], grad_norm=float(gnorm))
                run_loss.zero_()
                n_micro = 0
            if step % tcfg["eval_every"] == 0 or step == total:
                acc = run_eval(step, final=step == total)
                save("last", step)
                if acc > best:
                    best = acc
                    save("best", step)
                checkpoint(step)
            elif step % save_every == 0:
                checkpoint(step)
            if step >= total:
                break
    mark_done(out_dir, accelerator, f"best val acc {best:.4f}")
    if main:
        print(f"[done] best val acc {best:.4f} -> {out_dir}")
    accelerator.end_training()
    return out_dir
