"""过滤轨迹:结论正确,且引用的细胞 / 基因确实在输入中富集。

拒绝原因(可多条):
- format:缺少 reasoning / evidence / answer 或格式不对(先经 repair_format 修复可挽回的情况);
- wrong_answer:答案不是真值标签;
- no_evidence:evidence 为空;
- type_not_enriched / gene_not_enriched:evidence 中的细胞类型 / 基因在该邻域并不富集;
- unseen_gene:reasoning 里出现了面板中的基因,但它不在给教师看的富集基因列表里(用先验代替数据);
- leak:reasoning 透露了"标签是给定的"。
"""

from __future__ import annotations

import re

from spatial_reason.inference.generate import match_label, normalize_label, parse_output

LEAK = re.compile(r"\b(annotated label|given label|label (is|was) (given|provided)|we are told|as stated)\b", re.I)


def _norm_type(s: str) -> str:
    s = normalize_label(s)
    s = re.sub(r"\s*\(?\d+%.*$", "", s)  # 去掉 "tumor cells 60%" 之类的数字
    return re.sub(r"s$", "", s)


_PLACEHOLDER = re.compile(r"^<reasoning>\s*step-by-step reasoning grounded in the observed data\s*</reasoning>\s*(.+?)\s*(?=<evidence>)",
                          re.S | re.I)


def repair_format(text: str) -> str:
    """教师约 16% 的输出把格式模板里的占位符原样抄成 <reasoning>…</reasoning>,真正的推理写在标签外面。
    内容本身没问题,把标签外的推理移回 <reasoning> 里。"""
    m = _PLACEHOLDER.match(text.strip())
    if not m:
        return text
    return f"<reasoning>{m.group(1)}</reasoning>\n" + text.strip()[m.end():]


def check_trace(text: str, rec: dict, panel_genes: set[str]) -> tuple[bool, list[str], dict]:
    p = parse_output(repair_format(text))
    reasons = []
    if not p.well_formed or not p.reasoning:
        reasons.append("format")
    if match_label(p.answer, rec["candidates"]) != rec["label"]:
        reasons.append("wrong_answer")
    st = rec["stats"]
    if not (p.evidence_types or p.evidence_genes):
        reasons.append("no_evidence")
    enriched_t = {_norm_type(t) for t in st["enriched_types"]}
    if any(_norm_type(t) not in enriched_t for t in p.evidence_types):
        reasons.append("type_not_enriched")
    enriched_g = {g.upper() for g in st["enriched_genes"]}
    if any(g.upper() not in enriched_g for g in p.evidence_genes):
        reasons.append("gene_not_enriched")
    shown = {g.upper() for g in st["gene_effect"]}
    words = {w.strip(",.;:()") for w in re.split(r"[\s/]+", p.reasoning or "")}
    mentioned = {w for w in words if w in panel_genes}
    if any(w.upper() not in shown for w in mentioned):
        reasons.append("unseen_gene")
    if p.reasoning and LEAK.search(p.reasoning):
        reasons.append("leak")
    info = {"answer": p.answer, "evidence_types": p.evidence_types, "evidence_genes": p.evidence_genes,
            "mentioned_genes": sorted(mentioned)}
    return not reasons, reasons, info


def canonical_target(text: str, label: str) -> str:
    """把通过过滤的教师输出整理成学生的 SFT 目标:统一换行,答案写成候选标签的标准形式。"""
    text = repair_format(text)
    p = parse_output(text)
    ev = f"cell types: {'; '.join(p.evidence_types) or 'none'} | genes: {'; '.join(p.evidence_genes) or 'none'}"
    return f"<reasoning>{p.reasoning}</reasoning>\n<evidence>{ev}</evidence>\n<answer>{label}</answer>"
