"""SpatialLLM:Qwen3 (+ LoRA),把投影后的邻域嵌入作为 token 插入输入序列(参照 BioReason)。

做法与 BioReason `dna_llm.py` 相同:tokenizer 里加一个占位 token `<|cell|>`,
prompt 中每个细胞放一个占位符;前向时把这些位置的词嵌入替换成投影层的输出。
Qwen3 的嵌入矩阵比 tokenizer 词表大(151936 > 151669),新 token 不需要 resize。
"""

from __future__ import annotations

from pathlib import Path

import torch
from torch import nn

from spatial_reason.models.projector import CellProjector

CELL_TOKEN = "<|cell|>"


class SpatialLLM(nn.Module):
    def __init__(self, llm_path: str | Path, embed_dim: int, use_position: bool = True,
                 n_freqs: int = 8, freeze_llm: bool = True, dtype: torch.dtype = torch.bfloat16,
                 gradient_checkpointing: bool = False, device: str | torch.device = "cuda",
                 lora: dict | None = None, gene_encoder: dict | None = None):
        """lora:None 表示 LLM 完全冻结(阶段 0);给出 {r, alpha, dropout, target_modules} 时
        基座冻结、只训练 LoRA(阶段 1/2)。

        gene_encoder:方案 B(data.encoder=expr)时给出 {vocab, dim, hidden, text_layer},
        投影层前加可学习的基因集合编码器,基因名的文本表示取自本 LLM(见 gene_text_embeddings)。"""
        super().__init__()
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(llm_path)
        if CELL_TOKEN not in self.tokenizer.get_vocab():
            self.tokenizer.add_special_tokens({"additional_special_tokens": [CELL_TOKEN]})
        self.cell_token_id = self.tokenizer.convert_tokens_to_ids(CELL_TOKEN)

        self.llm = AutoModelForCausalLM.from_pretrained(llm_path, dtype=dtype).to(device)
        n_emb = self.llm.get_input_embeddings().weight.shape[0]
        if self.cell_token_id >= n_emb:
            self.llm.resize_token_embeddings(len(self.tokenizer))
        if freeze_llm or lora:
            self.llm.requires_grad_(False)
        if gradient_checkpointing:
            self.llm.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
            self.llm.config.use_cache = False
        if lora:
            from peft import LoraConfig, get_peft_model

            self.llm = get_peft_model(self.llm, LoraConfig(
                r=lora["r"], lora_alpha=lora["alpha"], lora_dropout=lora.get("dropout", 0.0),
                target_modules=lora["target_modules"], task_type="CAUSAL_LM"))
        # 只有"完全冻结、没有 LoRA"时才把 LLM 固定在 eval 模式;有 LoRA 时需要 dropout
        self.freeze_llm = freeze_llm and not lora
        if self.freeze_llm:
            self.llm.eval()

        hidden = self.llm.config.hidden_size
        gene_kw = {}
        if gene_encoder:
            from spatial_reason.encoders.expression import load_vocab

            vocab = load_vocab(gene_encoder.get("vocab", "data/models/expr/vocab.json"))
            gene_kw = {"gene_text": self.gene_text_embeddings(vocab, gene_encoder.get("text_layer", 24)),
                       "gene_dim": gene_encoder.get("dim", 256), "gene_hidden": gene_encoder.get("hidden", 512)}
        self.projector = CellProjector(embed_dim, hidden, use_position=use_position, n_freqs=n_freqs,
                                       **gene_kw).to(device)
        self._calibrate_scale()

    @torch.no_grad()
    def gene_text_embeddings(self, genes: list[str], layer: int = 24, batch_size: int = 128) -> torch.Tensor:
        """每个基因名在冻结 LLM 中的表示:prompt "The gene {symbol}" 最后一个 token 在第 layer 层的隐藏状态。

        用中间层而不是词嵌入:基因名常被切成多个子词("CD8A" -> "CD", "8", "A"),
        中间层的最后一个 token 已经把整个名字和 LLM 对它的知识聚合在一起。
        """
        base = self.llm.get_base_model() if hasattr(self.llm, "get_base_model") else self.llm
        dev = base.get_input_embeddings().weight.device
        side = self.tokenizer.padding_side
        self.tokenizer.padding_side = "right"  # 右侧 padding,位置编码与单独输入时一致
        try:
            out = []
            for i in range(0, len(genes), batch_size):
                enc = self.tokenizer([f"The gene {g}" for g in genes[i:i + batch_size]], return_tensors="pt",
                                     padding=True, add_special_tokens=False).to(dev)
                hs = base(**enc, output_hidden_states=True).hidden_states[layer]
                last = enc["attention_mask"].sum(1) - 1
                out.append(hs[torch.arange(len(last), device=dev), last].float().cpu())
        finally:
            self.tokenizer.padding_side = side
        return torch.cat(out)

    @property
    def device(self) -> torch.device:
        return self.projector.scale.device

    @torch.no_grad()
    def _calibrate_scale(self, n: int = 4096) -> None:
        """让投影输出的平均范数与普通文字 token 的嵌入范数一致。"""
        W = self.llm.get_input_embeddings().weight[: len(self.tokenizer) - 1].float()
        target = W.norm(dim=-1).median()
        in_dim = self.projector.norm.normalized_shape[0]
        x = torch.randn(n, in_dim, device=self.device)
        out = self.projector.mlp(self.projector.norm(x))
        self.projector.scale.fill_(float(target / out.norm(dim=-1).median()))

    def train(self, mode: bool = True):
        super().train(mode)
        if self.freeze_llm:
            self.llm.eval()
        return self

    def build_inputs_embeds(self, input_ids: torch.Tensor, cell_emb: torch.Tensor,
                            rel_pos: torch.Tensor | None) -> torch.Tensor:
        embeds = self.llm.get_input_embeddings()(input_ids)
        cells = self.projector(cell_emb.float(), None if rel_pos is None else rel_pos.float())
        mask = input_ids == self.cell_token_id
        n_slots, n_cells = int(mask.sum()), cells.shape[0] * cells.shape[1]
        if n_slots != n_cells:
            raise ValueError(f"占位符数 {n_slots} 与细胞 token 数 {n_cells} 不一致")
        # 布尔索引按行主序展开,与 cells 的 (B, N) 顺序一致
        embeds = embeds.masked_scatter(mask.unsqueeze(-1), cells.to(embeds.dtype).reshape(-1, embeds.shape[-1]))
        return embeds

    def forward(self, input_ids, attention_mask, cell_emb, rel_pos=None, labels=None):
        embeds = self.build_inputs_embeds(input_ids, cell_emb, rel_pos)
        return self.llm(inputs_embeds=embeds, attention_mask=attention_mask, labels=labels)

    @torch.no_grad()
    def generate(self, input_ids, attention_mask, cell_emb, rel_pos=None, **gen_kwargs):
        embeds = self.build_inputs_embeds(input_ids, cell_emb, rel_pos)
        use_cache = getattr(self.llm.config, "use_cache", True)
        self.llm.config.use_cache = True  # 梯度检查点会关掉 cache,生成时临时打开
        try:
            return self.llm.generate(inputs_embeds=embeds, attention_mask=attention_mask,
                                     pad_token_id=self.tokenizer.pad_token_id, use_cache=True, **gen_kwargs)
        finally:
            self.llm.config.use_cache = use_cache

    def save_lora(self, path: str | Path) -> None:
        if hasattr(self.llm, "peft_config"):
            self.llm.save_pretrained(str(path))

    def load_lora(self, path: str | Path) -> None:
        from peft import set_peft_model_state_dict
        from safetensors.torch import load_file

        set_peft_model_state_dict(self.llm, load_file(str(Path(path) / "adapter_model.safetensors")))

    def save_projector(self, path: str | Path, extra: dict | None = None) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save({"projector": self.projector.state_dict(), **(extra or {})}, path)

    def load_projector(self, path: str | Path) -> dict:
        ckpt = torch.load(path, map_location=self.device)
        self.projector.load_state_dict(ckpt["projector"])
        return ckpt
