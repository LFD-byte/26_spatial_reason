"""纯文本 LLM 基线:LLM 读同样的文本统计(组成、富集基因、候选标签),不给标签,自己推断。

    # 1. 构造请求(训练环境)
    python scripts/08_text_baseline.py build --dir data/traces/qwen3_32b
    # 2. 生成(vllm 环境,贪心解码)
    NCCL_P2P_DISABLE=1 ~/software/anaconda3/envs/vllm/bin/python scripts/05_run_teacher_vllm.py \
        --model data/models/Qwen3-32B --tp 2 --n 1 --temperature 0 --max-tokens 640 \
        --requests data/traces/qwen3_32b/text_only_val.jsonl --output data/traces/qwen3_32b/text_only_val_out.jsonl \
        --enforce-eager --disable-custom-all-reduce
    # 3. 打分,可选与 stage 1 学生对比
    python scripts/08_text_baseline.py score --dir data/traces/qwen3_32b \
        --compare outputs/stage1/novae_qwen3_32b_teacher/eval_full_best.json
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from spatial_reason.traces.filter import repair_format
from spatial_reason.traces.generate import text_only_request
from spatial_reason.training.stage1_sft import macro_f1, score_prediction


def build(d: Path, split: str) -> None:
    out = d / f"text_only_{split}.jsonl"
    n = 0
    with open(out, "w") as f:
        for line in open(d / f"requests_{split}.jsonl"):
            f.write(json.dumps(text_only_request(json.loads(line)), ensure_ascii=False) + "\n")
            n += 1
    print(f"[ok] {n} requests -> {out}")


def score(d: Path, split: str, compare: str | None, tag: str) -> None:
    reqs = {json.loads(l)["id"]: json.loads(l) for l in open(d / f"text_only_{split}.jsonl")}
    rows = []
    for line in open(d / f"text_only_{split}_out.jsonl"):
        t = json.loads(line)
        r = reqs[t["id"]]
        text = repair_format(t["outputs"][0])  # 教师偶尔把模板占位符原样抄进 <reasoning>,与过滤时同样修复
        rows.append({"id": r["id"], "dataset": r["dataset"], "label": r["label"], "generated": text,
                     **score_prediction(text, r)})
    by = defaultdict(list)
    for r in rows:
        by[r["dataset"]].append(r)
    res = {"n": len(rows)}
    for ds in sorted(by) + ["all"]:
        rs = rows if ds == "all" else by[ds]
        res[f"{ds}/acc"] = float(np.mean([r["correct"] for r in rs]))
        if ds != "all":
            res[f"{ds}/macro_f1"] = macro_f1([r["label"] for r in rs], [r["pred"] for r in rs])
        res[f"{ds}/format"] = float(np.mean([r["well_formed"] for r in rs]))
        ev = [r["evidence_valid"] for r in rs if r["n_cited"]]
        res[f"{ds}/evidence_valid"] = float(np.mean(ev)) if ev else 0.0
    (d / f"text_only_{split}_eval.json").write_text(json.dumps(res, indent=2))
    with open(d / f"text_only_{split}_scored.jsonl", "w") as f:
        f.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)

    other = json.loads(Path(compare).read_text()) if compare else {}
    print(f"{'dataset':22s} {tag + ' acc':>14s} {'f1':>6s} {'ev':>6s}" + ("   student acc     f1     ev" if other else ""))
    for ds in sorted(by) + ["all"]:
        line = f"{ds:22s} {res[f'{ds}/acc']:14.3f} {res.get(f'{ds}/macro_f1', float('nan')):6.3f} " \
               f"{res[f'{ds}/evidence_valid']:6.3f}"
        if other:
            line += f"   {other[f'{ds}/acc']:11.3f} {other.get(f'{ds}/macro_f1', float('nan')):6.3f} " \
                    f"{other[f'{ds}/evidence_valid']:6.3f}"
        print(line)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("cmd", choices=["build", "score"])
    p.add_argument("--dir", default="data/traces/qwen3_32b")
    p.add_argument("--split", default="val")
    p.add_argument("--compare", default=None, help="学生的 eval_full_*.json,用于并排对比")
    p.add_argument("--tag", default="text LLM")
    a = p.parse_args()
    (build if a.cmd == "build" else lambda d, s: score(d, s, a.compare, a.tag))(Path(a.dir), a.split)


if __name__ == "__main__":
    main()
