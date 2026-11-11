"""批量生成与输出解析(推理链 / 证据 / 答案)。

统一输出格式(教师轨迹、学生 SFT 目标、GRPO 格式奖励都用这一种):

    <reasoning>...</reasoning>
    <evidence>cell types: A; B | genes: X; Y</evidence>
    <answer>标签</answer>
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

OUTPUT_FORMAT = (
    "<reasoning>step-by-step reasoning grounded in the observed data</reasoning>\n"
    "<evidence>cell types: <enriched cell types you relied on, separated by ;> | "
    "genes: <enriched genes you relied on, separated by ;></evidence>\n"
    "<answer><exactly one label from the candidate list></answer>"
)

_TAG = {t: re.compile(rf"<{t}>(.*?)</{t}>", re.S | re.I) for t in ("reasoning", "evidence", "answer")}


@dataclass
class ParsedOutput:
    reasoning: str | None = None
    evidence_types: list[str] = field(default_factory=list)
    evidence_genes: list[str] = field(default_factory=list)
    answer: str | None = None
    well_formed: bool = False


def _split(s: str) -> list[str]:
    return [x.strip().strip(".") for x in re.split(r"[;,]", s) if x.strip().strip(".") and
            x.strip().lower() not in {"none", "n/a", "-"}]


def parse_output(text: str) -> ParsedOutput:
    out = ParsedOutput()
    m = {t: r.findall(text) for t, r in _TAG.items()}
    if m["reasoning"]:
        out.reasoning = m["reasoning"][0].strip()
    if m["answer"]:
        out.answer = m["answer"][-1].strip()
    if m["evidence"]:
        ev = m["evidence"][0]
        tm = re.search(r"cell types?\s*:\s*(.*?)(?:\||$)", ev, re.I | re.S)
        gm = re.search(r"genes?\s*:\s*(.*)$", ev, re.I | re.S)
        out.evidence_types = _split(tm.group(1)) if tm else []
        out.evidence_genes = _split(gm.group(1)) if gm else []
    # 每个标签恰好一次、顺序正确,且标签外没有多余文字
    once = all(len(m[t]) == 1 for t in _TAG)
    order = once and text.lower().find("<reasoning>") < text.lower().find("<evidence>") < text.lower().find("<answer>")
    stripped = re.sub(r"<(reasoning|evidence|answer)>.*?</\1>", "", text, flags=re.S | re.I).strip()
    out.well_formed = bool(order and not stripped)
    return out


def normalize_label(s: str | None) -> str:
    return re.sub(r"\s+", " ", (s or "").strip().strip(".").lower())


def match_label(answer: str | None, candidates: list[str]) -> str | None:
    """答案与候选标签匹配:先精确(忽略大小写),再接受 MERFISH 缩写形式(如只写 "PVH")。"""
    a = normalize_label(answer)
    for c in candidates:
        if a == normalize_label(c):
            return c
    for c in candidates:
        abbr = normalize_label(c.split(" (")[0])
        if a == abbr:
            return c
    return None
