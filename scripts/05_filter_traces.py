"""阶段 1 第 3 步:过滤教师轨迹(训练环境)。

    python scripts/05_filter_traces.py --dir data/traces/qwen3_32b
每个邻域最多保留 1 条通过的轨迹;输出 sft_{train,val}.jsonl 和拒绝原因统计 filter_report.json。
"""

import argparse
import json
from collections import Counter
from pathlib import Path

from spatial_reason.traces.filter import canonical_target, check_trace


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dir", default="data/traces/qwen3_32b")
    args = p.parse_args()
    d = Path(args.dir)
    panels = {f.stem.removeprefix("panel_genes_"): set(json.loads(f.read_text())) for f in d.glob("panel_genes_*.json")}
    report = {}
    for split in ("train", "val"):
        tpath = d / f"teacher_{split}.jsonl"
        if not tpath.exists():
            print(f"[skip] {tpath} 不存在")
            continue
        reqs = {json.loads(l)["id"]: json.loads(l) for l in open(d / f"requests_{split}.jsonl")}
        reasons, kept, n_out = Counter(), [], 0
        per_ds = Counter()
        per_ds_kept = Counter()
        for line in open(tpath):
            t = json.loads(line)
            rec = reqs[t["id"]]
            per_ds[rec["dataset"]] += 1
            chosen = None
            for text in t["outputs"]:
                n_out += 1
                ok, why, info = check_trace(text, rec, panels[rec["dataset"]])
                reasons.update(why)
                if ok and chosen is None:
                    chosen = (text, info)
            if chosen:
                per_ds_kept[rec["dataset"]] += 1
                kept.append({k: rec[k] for k in ("id", "dataset", "sample", "center", "label", "candidates")}
                            | {"target": canonical_target(chosen[0], rec["label"]), "stats": rec["stats"],
                               "evidence_types": chosen[1]["evidence_types"],
                               "evidence_genes": chosen[1]["evidence_genes"]})
        with open(d / f"sft_{split}.jsonl", "w") as f:
            f.writelines(json.dumps(k, ensure_ascii=False) + "\n" for k in kept)
        labels = Counter((k["dataset"], k["label"]) for k in kept)
        report[split] = {"neighborhoods": len(reqs), "outputs": n_out, "kept": len(kept),
                         "keep_rate_by_dataset": {ds: round(per_ds_kept[ds] / per_ds[ds], 3) for ds in per_ds},
                         "reject_reasons_per_output": {k: round(v / n_out, 3) for k, v in reasons.most_common()},
                         "kept_per_label": {f"{a}|{b}": c for (a, b), c in sorted(labels.items())}}
        print(f"[{split}] kept {len(kept)}/{len(reqs)}; reasons {dict(reasons.most_common())}")
    (d / "filter_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
