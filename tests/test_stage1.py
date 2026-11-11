"""阶段 1 组件:输出解析、标签匹配、轨迹过滤、LoRA 注入。"""

from pathlib import Path

import pytest
import torch

from spatial_reason.inference.generate import match_label, parse_output
from spatial_reason.traces.filter import canonical_target, check_trace

QWEN = Path("data/models/Qwen3-4B")
needs_qwen = pytest.mark.skipif(not (QWEN / "tokenizer.json").exists(), reason="需要 Qwen3-4B tokenizer")

REC = {
    "label": "tumor-stroma boundary",
    "candidates": ["stroma", "tumor interior", "tumor-stroma boundary"],
    "stats": {"enriched_types": ["tumor cells", "fibroblasts"], "enriched_genes": ["COL1A1", "KRT7", "MMP11"],
              "gene_effect": {"COL1A1": 1.2, "KRT7": 0.9}},
}
PANEL = {"COL1A1", "KRT7", "MMP11", "CD3E", "EPCAM"}
GOOD = ("<reasoning>Tumor cells are 55% and fibroblasts 30%, with COL1A1 and KRT7 both up, so the "
        "neighborhood mixes epithelium and stroma.</reasoning>\n"
        "<evidence>cell types: tumor cells; fibroblasts | genes: COL1A1; KRT7</evidence>\n"
        "<answer>tumor-stroma boundary</answer>")


def test_parse_output_well_formed():
    p = parse_output(GOOD)
    assert p.well_formed and p.answer == "tumor-stroma boundary"
    assert p.evidence_types == ["tumor cells", "fibroblasts"] and p.evidence_genes == ["COL1A1", "KRT7"]
    assert not parse_output("Answer: stroma").well_formed
    assert not parse_output(GOOD + " extra text").well_formed


def test_match_label_accepts_abbreviation():
    cands = ["PVH (paraventricular hypothalamic nucleus)", "PV (periventricular hypothalamic nucleus)"]
    assert match_label("PVH", cands) == cands[0]
    assert match_label("pv (periventricular hypothalamic nucleus).", cands) == cands[1]
    assert match_label("thalamus", cands) is None


def test_filter_accepts_good_trace():
    ok, why, _ = check_trace(GOOD, REC, PANEL)
    assert ok, why
    assert canonical_target(GOOD, REC["label"]).endswith("<answer>tumor-stroma boundary</answer>")


@pytest.mark.parametrize("text,reason", [
    (GOOD.replace("<answer>tumor-stroma boundary", "<answer>stroma"), "wrong_answer"),
    (GOOD.replace("genes: COL1A1; KRT7", "genes: COL1A1; CD3E"), "gene_not_enriched"),
    (GOOD.replace("cell types: tumor cells;", "cell types: T cells;"), "type_not_enriched"),
    (GOOD.replace("KRT7 both up", "KRT7 both up and EPCAM"), "unseen_gene"),
    (GOOD.replace("Tumor cells", "Since the annotated label says so, tumor cells"), "leak"),
    ("<answer>tumor-stroma boundary</answer>", "format"),
])
def test_filter_rejects(text, reason):
    ok, why, _ = check_trace(text, REC, PANEL)
    assert not ok and reason in why


@needs_qwen
def test_lora_only_lora_and_projector_train(tmp_path):
    from transformers import AutoTokenizer, Qwen3Config, Qwen3ForCausalLM

    from spatial_reason.data.collate import Collator
    from spatial_reason.models.spatial_llm import CELL_TOKEN, SpatialLLM

    cfg = Qwen3Config(vocab_size=151936, hidden_size=64, intermediate_size=128, num_hidden_layers=2,
                      num_attention_heads=4, num_key_value_heads=2, head_dim=16, tie_word_embeddings=True)
    Qwen3ForCausalLM(cfg).save_pretrained(tmp_path)
    AutoTokenizer.from_pretrained(QWEN).save_pretrained(tmp_path)
    m = SpatialLLM(tmp_path, embed_dim=8, device="cpu", dtype=torch.float32, gradient_checkpointing=True,
                   lora={"r": 4, "alpha": 8, "dropout": 0.0, "target_modules": ["q_proj", "v_proj"]})
    batch = [{"cell_emb": torch.randn(3, 8), "rel_pos": torch.randn(3, 2), "prompt": "x " + CELL_TOKEN * 3,
              "target": "<answer>a</answer>", "meta": {}} for _ in range(2)]
    b = Collator(m.tokenizer)(batch)
    m.train()
    m(b["input_ids"], b["attention_mask"], b["cell_emb"], b["rel_pos"], labels=b["labels"]).loss.backward()
    named = dict(m.llm.named_parameters())
    assert all(p.grad is not None for n, p in named.items() if "lora_" in n)
    assert all(p.grad is None for n, p in named.items() if "lora_" not in n)
    assert all(p.grad is not None for p in m.projector.parameters())
    m.save_lora(tmp_path / "lora")
    assert (tmp_path / "lora" / "adapter_model.safetensors").exists()
    out = m.generate(b["input_ids"], b["attention_mask"], b["cell_emb"], b["rel_pos"], max_new_tokens=3)
    assert out.shape[0] == 2


def test_filter_repairs_copied_placeholder():
    body = GOOD.split("<reasoning>")[1].split("</reasoning>")[0]
    bad = ("<reasoning>step-by-step reasoning grounded in the observed data</reasoning>  \n" + body + "\n\n"
           + GOOD.split("</reasoning>\n")[1])
    assert not parse_output(bad).well_formed
    ok, why, _ = check_trace(bad, REC, PANEL)
    assert ok, why
    assert parse_output(canonical_target(bad, REC["label"])).reasoning == body
