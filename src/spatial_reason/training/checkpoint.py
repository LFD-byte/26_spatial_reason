"""断点续训:定期保存可训练参数 + 优化器 + 调度器 + 步数,停机后自动接着训。

只存 requires_grad 的参数(投影层 / LoRA),冻结的 LLM 从原始权重重新加载,
所以 checkpoint 只有几百 MB,而不是 accelerator.save_state 那样整模型 8 GB。

写入是原子的:先写 checkpoint.pt.tmp 再 os.replace,停机时最多丢掉正在写的那一份,
不会留下损坏文件。

数据顺序不做逐 batch 复现:两个阶段都是有放回的加权随机采样,续训时用 seed + step
重新设种子继续抽样,统计上与不中断等价,且不需要记录采样器内部状态。
"""

from __future__ import annotations

import os
from pathlib import Path

import torch

from spatial_reason.utils.seed import set_seed

CKPT_NAME = "checkpoint.pt"
DONE_NAME = "DONE"


def trainable_state(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {n: p.detach().cpu().clone() for n, p in model.named_parameters() if p.requires_grad}


def save_checkpoint(out_dir: Path, accelerator, model, opt, sched, step: int, state: dict) -> None:
    """所有进程都要调用(内部同步);只有主进程写盘。"""
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        path = Path(out_dir) / CKPT_NAME
        tmp = path.with_suffix(".pt.tmp")
        torch.save({"step": step, "params": trainable_state(model), "optimizer": opt.state_dict(),
                    "scheduler": sched.state_dict(), "state": state}, tmp)
        os.replace(tmp, path)
    accelerator.wait_for_everyone()


def load_checkpoint(out_dir: Path, model, opt, sched, seed: int) -> tuple[int, dict] | None:
    """有 checkpoint 就恢复,返回 (step, state);否则返回 None。每个进程各自读。"""
    path = Path(out_dir) / CKPT_NAME
    if not path.exists():
        return None
    ck = torch.load(path, map_location="cpu", weights_only=False)
    named = dict(model.named_parameters())
    missing = [n for n in ck["params"] if n not in named]
    if missing:
        raise RuntimeError(f"checkpoint 参数与模型不匹配(配置变了?):{missing[:3]}")
    with torch.no_grad():
        for n, v in ck["params"].items():
            named[n].copy_(v.to(named[n].device, named[n].dtype))
    opt.load_state_dict(ck["optimizer"])
    sched.load_state_dict(ck["scheduler"])
    set_seed(seed + ck["step"])  # 续训后的采样流与中断前不同,但同分布
    return ck["step"], ck["state"]


def mark_done(out_dir: Path, accelerator, info: str = "") -> None:
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        (Path(out_dir) / DONE_NAME).write_text(info + "\n")


def is_done(out_dir: Path) -> bool:
    return (Path(out_dir) / DONE_NAME).exists()
