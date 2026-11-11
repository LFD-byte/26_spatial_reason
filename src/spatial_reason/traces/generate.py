"""教师 LLM 读取文本形式的数据与真值标签,生成推理轨迹。

流程分三步(教师在单独的 vllm 环境里跑,不依赖本包):
1. build_requests:在训练环境里为抽样的中心细胞构造教师 prompt,写 JSONL(含元信息与富集统计);
2. scripts/05_run_teacher_vllm.py:vllm 环境读 JSONL 生成,写回 JSONL;
3. traces.filter:只保留结论正确、引用确实富集的轨迹。

教师看到的是文本(组成、富集基因、候选标签)和真值标签;学生之后看到的是嵌入,学同样的轨迹。
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from spatial_reason.data.collate import SliceData, load_slice
from spatial_reason.data.datasets import DATASET_INFO, domain_candidates, domain_name
from spatial_reason.data.splits import split_samples
from spatial_reason.evaluation.enrichment import EnrichmentCalculator, EnrichmentCriteria, NeighborhoodStats
from spatial_reason.inference.generate import OUTPUT_FORMAT

TEACHER_SYSTEM = (
    "You are an expert in spatial transcriptomics and tissue biology. You explain how the identity of a "
    "tissue region follows from the measured cellular composition and gene expression of a cell "
    "neighborhood. You only use evidence that is present in the data you are given."
)

TEACHER_USER = """A cell neighborhood (a center cell and its {k} nearest neighbors) from a {platform} section of {tissue}.

Candidate labels:
{candidates}

Observed data:
{data}

The annotated label of this neighborhood is: {label}

Write the reasoning that a careful analyst would follow to reach this label from the observed data alone.
Rules:
- Cite only cell types and genes that appear in the observed data above, with their numbers where useful.
- Explain why the label fits better than the most plausible alternative candidates.
- Do not say or imply that the label was given to you; reason as if you were inferring it.
- Keep the reasoning to 3-6 sentences.
- In <evidence>, list only cell types and genes that are enriched in this neighborhood (listed above).

Answer in exactly this format and nothing else:
{fmt}"""


def neighborhood_text(st: NeighborhoodStats, has_cell_type: bool) -> str:
    """把富集统计写成文本(教师输入;也用于纯文本 Qwen 基线)。"""
    lines = []
    if has_cell_type:
        lines.append(f"- Center cell type: {st.center_type}")
        comp = "; ".join(f"{n} {100 * f:.0f}% (section-wide {100 * st.slice_composition[n]:.0f}%)"
                         for n, f in st.composition.items())
        lines.append(f"- Cell-type composition of the neighborhood: {comp}")
        # 明确哪些类型算"富集",否则教师常把组成表里未富集的类型写进 <evidence>,被过滤掉
        enr = "; ".join(st.enriched_types) if st.enriched_types else "none"
        lines.append(f"- Cell types enriched in the neighborhood (the only ones allowed in <evidence>): {enr}")
    if st.gene_effect:
        genes = "; ".join(f"{g} (+{e:.1f} SD)" for g, e in st.gene_effect.items())
        lines.append(f"- Genes enriched in the neighborhood relative to the whole section "
                     f"(effect size in section-level SD units): {genes}")
    else:
        lines.append("- Genes enriched in the neighborhood relative to the whole section: none")
    return "\n".join(lines)


def format_candidates(cands: list[str]) -> str:
    return "\n".join(f"- {c}" for c in cands)


def class_balanced_centers(s: SliceData, dataset: str, per_label: int, rng: np.random.Generator,
                           ) -> list[int]:
    labels = np.array([domain_name(dataset, v) for v in s.domain], dtype=object)
    out = []
    for lab in sorted({x for x in labels if x is not None}):
        idx = np.where(labels == lab)[0]
        out.extend(rng.choice(idx, min(per_label, len(idx)), replace=False).tolist())
    return sorted(out)


def build_requests(cfg: dict, split: str, out_path: Path | str) -> int:
    """为 split(train/val)构造教师请求。cfg 结构见 configs/train/stage1_sft.yaml 的 traces 段。"""
    tcfg = cfg["traces"]
    crit = EnrichmentCriteria(**tcfg.get("criteria", {}))
    rng = np.random.default_rng(tcfg.get("seed", 0) + (0 if split == "train" else 1))
    k = cfg["data"]["k_neighbors"]
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(out_path, "w") as f:
        for ds, opts in cfg["data"]["datasets"].items():
            samples = split_samples(ds, opts.get("val_samples", []))[split]
            per_label = opts["traces_per_label_" + split]
            # 候选标签取该数据集所有切片的并集,保证 train/val 一致
            all_slices = {s: load_slice(ds, s, cfg["data"]["encoder"], k)
                          for s in split_samples(ds, opts.get("val_samples", []))["train"] + samples}
            cands = domain_candidates(ds, np.concatenate([sl.domain for sl in all_slices.values()]))
            # 每张切片分到的配额:per_label 平均分给各切片
            per_slice = max(1, int(np.ceil(per_label / len(samples))))
            info = DATASET_INFO[ds]
            for sname in samples:
                s = all_slices[sname]
                calc = EnrichmentCalculator(s.desc, crit)
                for c in class_balanced_centers(s, ds, per_slice, rng):
                    st = calc.stats(s.nb.index[c], top_genes=tcfg.get("top_genes", 10))
                    label = domain_name(ds, s.domain[c])
                    user = TEACHER_USER.format(
                        k=k, platform=info["platform"], tissue=info["tissue"],
                        candidates=format_candidates(cands), data=neighborhood_text(st, s.desc.has_cell_type),
                        label=label, fmt=OUTPUT_FORMAT)
                    rec = {"id": f"{ds}|{sname}|{c}", "dataset": ds, "sample": sname, "center": int(c),
                           "label": label, "candidates": cands, "stats": st.to_dict(),
                           "messages": [{"role": "system", "content": TEACHER_SYSTEM},
                                        {"role": "user", "content": user}]}
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    n += 1
            # 面板基因表单独存一份(过滤时检查"提到了输入里没有的基因")
            panel = out_path.with_name(f"panel_genes_{ds}.json")
            panel.write_text(json.dumps(sorted(map(str, s.desc.genes))))
    return n
