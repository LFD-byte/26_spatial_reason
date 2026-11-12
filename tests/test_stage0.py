"""阶段 0 各组件的单元测试。不加载 4B 模型:用随机初始化的极小 Qwen3 + 真实 tokenizer。"""

from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import pytest
import torch

from spatial_reason.data.collate import Collator
from spatial_reason.data.descriptions import DescriptionBuilder, display_name
from spatial_reason.data.neighborhoods import build_neighborhoods
from spatial_reason.models.position import FourierPosition
from spatial_reason.models.projector import CellProjector
from spatial_reason.training.stage0_align import score_description

QWEN = Path("data/models/Qwen3-4B")
needs_qwen = pytest.mark.skipif(not (QWEN / "tokenizer.json").exists(), reason="需要 Qwen3-4B tokenizer")


def toy_adata(n=200, seed=0):
    rng = np.random.default_rng(seed)
    xy = rng.uniform(0, 100, size=(n, 2))
    left = xy[:, 0] < 50
    X = rng.poisson(1.0, size=(n, 5)).astype(np.float32)
    X[left, 0] += 20  # GENE_A 只在左半边高表达
    obs = pd.DataFrame({"cell_type": np.where(left, "fibroblast", "macrophage"),
                        "domain": np.where(left, "L", "R")}, index=[f"c{i}" for i in range(n)])
    a = ad.AnnData(X=X, obs=obs, var=pd.DataFrame(index=["GENE_A", "GENE_B", "GENE_C", "GENE_D", "NegPrb1"]))
    a.obsm["spatial"] = xy
    return a


def test_neighborhoods_center_first_and_relative():
    a = toy_adata()
    nb = build_neighborhoods(a.obsm["spatial"], k=8)
    assert nb.index.shape == (200, 9)
    assert (nb.index[:, 0] == np.arange(200)).all()
    assert np.allclose(nb.rel_pos[:, 0], 0)
    d = np.linalg.norm(nb.rel_pos, axis=-1)
    assert (np.diff(d, axis=1) >= -1e-6).all()  # 按距离升序


def test_description_composition_and_genes():
    a = toy_adata()
    nb = build_neighborhoods(a.obsm["spatial"], k=16)
    b = DescriptionBuilder(a, "cosmx_nsclc")
    assert "NegPrb1" not in b.genes
    i = int(np.argmin(np.abs(a.obsm["spatial"][:, 0] - 10)))  # 左侧深处
    d = b.describe(nb.index[i], template=0)
    assert d.cell_types[0] == ("fibroblasts", 100)
    assert d.genes == ["GENE_A"]
    assert "100% fibroblasts" in d.text and "GENE_A" in d.text
    assert "L" not in d.text.split()  # 不泄漏 domain 标签


def test_soft_composition_from_deconvolution():
    """Visium:obsm['cell_type_props'] 存在时,组成取各 spot 比例的平均,中心类型取比例最高者。"""
    from spatial_reason.evaluation.enrichment import EnrichmentCalculator

    a = toy_adata()
    left = a.obsm["spatial"][:, 0] < 50
    a.obs["cell_type"] = None
    a.obsm["cell_type_props"] = pd.DataFrame(
        {"Excit_L4": np.where(left, 0.7, 0.1), "Oligo": np.where(left, 0.3, 0.9)}, index=a.obs_names)
    nb = build_neighborhoods(a.obsm["spatial"], k=16)
    b = DescriptionBuilder(a, "dlpfc")
    assert b.soft and b.has_cell_type
    i = int(np.argmin(np.abs(a.obsm["spatial"][:, 0] - 10)))
    assert b.describe(nb.index[i], template=0).cell_types == [("L4 excitatory neurons", 70),
                                                              ("oligodendrocytes", 30)]
    st = EnrichmentCalculator(b).stats(nb.index[i])
    assert st.center_type == "L4 excitatory neurons"
    assert st.enriched_types == ["L4 excitatory neurons"]  # 0.7 vs 切片约 0.4,比值 >= 1.5


def test_display_names():
    assert display_name("Excit_L3_4_5", "dlpfc") == "L3/4/5 excitatory neurons"
    assert display_name("tumor 9", "cosmx_nsclc") == "tumor cells"
    assert display_name("Endothelial 2", "merfish_hypothalamus") == "endothelial cells"
    assert display_name("eL2/3", "starmap") == "L2/3 excitatory neurons"


def test_projector_shapes_and_position():
    pos = FourierPosition(n_freqs=4)
    assert pos(torch.zeros(3, 7, 2)).shape == (3, 7, pos.out_dim)
    proj = CellProjector(embed_dim=64, llm_dim=32)
    out = proj(torch.randn(2, 33, 64), torch.randn(2, 33, 2))
    assert out.shape == (2, 33, 32)


def test_score_description():
    meta = {"cell_types": [("tumor cells", 60), ("fibroblasts", 20)], "genes": ["VEGFA", "SLC2A1"],
            "center_type": "tumor cells"}
    text = "Composition: 55% tumor cells and 25% fibroblasts. Center cell: tumor cell. Enriched genes: VEGFA, MKI67."
    s = score_description(text, meta, {"VEGFA", "SLC2A1", "MKI67"})
    assert s["type_f1"] == 1.0 and s["top_type_acc"] == 1.0 and s["pct_mae"] == 5.0
    assert s["gene_precision"] == 0.5 and s["gene_recall"] == 0.5


@pytest.fixture(scope="module")
def tiny_llm(tmp_path_factory):
    from transformers import AutoTokenizer, Qwen3Config, Qwen3ForCausalLM

    d = tmp_path_factory.mktemp("tiny_qwen")
    cfg = Qwen3Config(vocab_size=151936, hidden_size=64, intermediate_size=128, num_hidden_layers=2,
                      num_attention_heads=4, num_key_value_heads=2, head_dim=16, tie_word_embeddings=True)
    torch.manual_seed(0)
    Qwen3ForCausalLM(cfg).save_pretrained(d)
    AutoTokenizer.from_pretrained(QWEN).save_pretrained(d)
    return d


@needs_qwen
def test_collator_masks_prompt(tiny_llm):
    from transformers import AutoTokenizer

    from spatial_reason.models.spatial_llm import CELL_TOKEN

    tok = AutoTokenizer.from_pretrained(tiny_llm)
    tok.add_special_tokens({"additional_special_tokens": [CELL_TOKEN]})
    cid = tok.convert_tokens_to_ids(CELL_TOKEN)
    batch = [{"cell_emb": torch.zeros(5, 4), "rel_pos": torch.zeros(5, 2), "prompt": "x " + CELL_TOKEN * 5,
              "target": t, "meta": {}} for t in ["short", "a much longer target sentence"]]
    out = Collator(tok)(batch)
    assert ((out["input_ids"] == cid).sum(1) == 5).all()
    for i, t in enumerate(["short", "a much longer target sentence"]):
        lab = out["labels"][i][out["labels"][i] != -100]
        assert tok.decode(lab) == t + "<|im_end|>"
    assert (out["labels"][out["attention_mask"] == 0] == -100).all()


@needs_qwen
def test_spatial_llm_injects_and_only_projector_trains(tiny_llm):
    from spatial_reason.models.spatial_llm import CELL_TOKEN, SpatialLLM

    m = SpatialLLM(tiny_llm, embed_dim=8, device="cpu", dtype=torch.float32)
    batch = [{"cell_emb": torch.randn(5, 8), "rel_pos": torch.randn(5, 2), "prompt": "x " + CELL_TOKEN * 5,
              "target": "y", "meta": {}} for _ in range(2)]
    b = Collator(m.tokenizer)(batch)
    emb = m.build_inputs_embeds(b["input_ids"], b["cell_emb"], b["rel_pos"])
    mask = b["input_ids"] == m.cell_token_id
    expect = m.projector(b["cell_emb"], b["rel_pos"]).reshape(-1, emb.shape[-1])
    assert torch.allclose(emb[mask], expect, atol=1e-6)

    out = m(b["input_ids"], b["attention_mask"], b["cell_emb"], b["rel_pos"], labels=b["labels"])
    out.loss.backward()
    assert all(p.grad is None for p in m.llm.parameters())
    assert all(p.grad is not None for p in m.projector.parameters())

    b2 = dict(b, cell_emb=b["cell_emb"][:1])  # 占位符数与细胞数不符应报错
    with pytest.raises(ValueError):
        m.build_inputs_embeds(b2["input_ids"], b2["cell_emb"], b2["rel_pos"][:1])


def test_gene_set_encoder_ignores_unmeasured_genes():
    from spatial_reason.models.projector import GeneSetEncoder

    enc = GeneSetEncoder(torch.randn(10, 16), dim=8, hidden=12)
    assert torch.all(enc.residual == 0)  # 残差零初始化:初始基因嵌入完全由基因名决定
    x = torch.randn(2, 5, 10)
    x[..., 7:] = 0  # 这个 panel 没测后 3 个基因
    out = enc(x)
    assert out.shape == (2, 5, 8)
    with torch.no_grad():
        enc.residual[7:] += 5.0  # 没测的基因的嵌入不影响输出
    assert torch.allclose(enc(x), out, atol=1e-5)
    proj = CellProjector(embed_dim=10, llm_dim=32, gene_text=torch.randn(10, 16), gene_dim=8)
    assert proj(x, torch.randn(2, 5, 2)).shape == (2, 5, 32)
    with pytest.raises(ValueError):
        CellProjector(embed_dim=11, llm_dim=32, gene_text=torch.randn(10, 16))


def test_spatial_llm_gene_encoder_trains_end_to_end(tiny_llm, tmp_path):
    import json

    from spatial_reason.models.spatial_llm import CELL_TOKEN, SpatialLLM

    vocab = tmp_path / "vocab.json"
    vocab.write_text(json.dumps({"genes": ["CD8A", "GAD1", "PTPRC", "MBP"]}))
    m = SpatialLLM(tiny_llm, embed_dim=4, device="cpu", dtype=torch.float32,
                   gene_encoder={"vocab": str(vocab), "dim": 8, "hidden": 12, "text_layer": 1})
    t = m.projector.gene_encoder.text
    assert t.shape[0] == 4 and not torch.allclose(t[0], t[1])  # 不同基因名的文本表示不同
    batch = [{"cell_emb": torch.randn(5, 4), "rel_pos": torch.randn(5, 2), "prompt": "x " + CELL_TOKEN * 5,
              "target": "y", "meta": {}} for _ in range(2)]
    b = Collator(m.tokenizer)(batch)
    m(b["input_ids"], b["attention_mask"], b["cell_emb"], b["rel_pos"], labels=b["labels"]).loss.backward()
    assert all(p.grad is None for p in m.llm.parameters())
    assert all(p.grad is not None for p in m.projector.gene_encoder.parameters())
    assert "gene_encoder.text" not in m.projector.state_dict()  # 文本表示不存盘,构造时重算
