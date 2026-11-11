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
                 lora: dict | None = None):
        """lora:None 表示 LLM 完全冻结(阶段 0);给出 {r, alpha, dropout, target_modules} 时
        基座冻结、只训练 LoRA(阶段 1/2)。"""
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
        self.projector = CellProjector(embed_dim, hidden, use_position=use_position, n_freqs=n_freqs).to(device)
        self._calibrate_scale()

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
