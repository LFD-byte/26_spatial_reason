"""两层 MLP,把编码器嵌入(+ 相对坐标特征)映射到 LLM 隐藏维度。

输入先做 LayerNorm(不同编码器的嵌入尺度差别很大),输出再乘一个可学习的标量,
初始化为 LLM 词嵌入的典型范数比例,让细胞 token 和文字 token 的尺度一开始就接近。

方案 B(data.encoder=expr):投影层前面多一个可学习的 GeneSetEncoder,直接读共享基因空间里的
标准化表达,与投影层一起端到端训练(阶段 0 起就可训练)。
"""

from __future__ import annotations

import torch
from torch import nn

from spatial_reason.models.position import FourierPosition


class GeneSetEncoder(nn.Module):
    """x (..., V) 标准化表达 -> (..., dim)。

    细胞表示 = Σ_g x_g · e_g,基因嵌入 e_g = A · t_g + r_g:
    - t_g:基因名在冻结 LLM 里的文本表示(buffer,不训练、不存盘,构造时重新计算);
    - A:所有基因共享的线性映射;r_g:每个基因的残差,零初始化。
    没见过的基因(新 panel)r_g = 0,仍能通过基因名得到嵌入,所以编码器不绑定某个 panel。
    """

    def __init__(self, gene_text: torch.Tensor, dim: int = 256, hidden: int = 512):
        super().__init__()
        t = gene_text.float()
        t = (t - t.mean(0)) / (t.std(0) + 1e-6)  # 各维标准化,A 的初始化尺度与 LLM 无关
        self.register_buffer("text", t, persistent=False)
        self.text_proj = nn.Linear(t.shape[1], hidden, bias=False)
        nn.init.normal_(self.text_proj.weight, std=t.shape[1] ** -0.5)
        self.residual = nn.Parameter(torch.zeros(t.shape[0], hidden))
        self.norm = nn.LayerNorm(hidden)
        self.out = nn.Sequential(nn.GELU(), nn.Linear(hidden, dim))
        self.out_dim = dim

    def gene_embeddings(self) -> torch.Tensor:
        return self.text_proj(self.text) + self.residual  # (V, hidden)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = x.float() @ self.gene_embeddings()
        return self.out(self.norm(h))


class CellProjector(nn.Module):
    def __init__(self, embed_dim: int, llm_dim: int, use_position: bool = True,
                 n_freqs: int = 8, output_scale: float = 1.0, gene_text: torch.Tensor | None = None,
                 gene_dim: int = 256, gene_hidden: int = 512):
        super().__init__()
        self.gene_encoder = None
        if gene_text is not None:
            if gene_text.shape[0] != embed_dim:
                raise ValueError(f"基因表 {gene_text.shape[0]} 个基因,但输入维度为 {embed_dim}")
            self.gene_encoder = GeneSetEncoder(gene_text, gene_dim, gene_hidden)
            embed_dim = gene_dim
        self.use_position = use_position
        self.pos = FourierPosition(n_freqs) if use_position else None
        in_dim = embed_dim + (self.pos.out_dim if use_position else 0)
        self.norm = nn.LayerNorm(in_dim)
        self.mlp = nn.Sequential(nn.Linear(in_dim, llm_dim), nn.GELU(), nn.Linear(llm_dim, llm_dim))
        self.scale = nn.Parameter(torch.tensor(float(output_scale)))

    def encode_cells(self, cell_emb: torch.Tensor) -> torch.Tensor:
        """投影前的细胞表示(方案 B 时为基因集合编码器的输出),也用于线性探针。"""
        return self.gene_encoder(cell_emb) if self.gene_encoder is not None else cell_emb

    def forward(self, cell_emb: torch.Tensor, rel_pos: torch.Tensor | None = None) -> torch.Tensor:
        """cell_emb: (B, N, D), rel_pos: (B, N, 2) -> (B, N, llm_dim)"""
        x = self.encode_cells(cell_emb)
        if self.use_position:
            x = torch.cat([x, self.pos(rel_pos)], dim=-1)
        return self.mlp(self.norm(x)) * self.scale
