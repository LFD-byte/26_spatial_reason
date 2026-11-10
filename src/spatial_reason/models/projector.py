"""两层 MLP,把编码器嵌入(+ 相对坐标特征)映射到 LLM 隐藏维度。

输入先做 LayerNorm(不同编码器的嵌入尺度差别很大),输出再乘一个可学习的标量,
初始化为 LLM 词嵌入的典型范数比例,让细胞 token 和文字 token 的尺度一开始就接近。
"""

from __future__ import annotations

import torch
from torch import nn

from spatial_reason.models.position import FourierPosition


class CellProjector(nn.Module):
    def __init__(self, embed_dim: int, llm_dim: int, use_position: bool = True,
                 n_freqs: int = 8, output_scale: float = 1.0):
        super().__init__()
        self.use_position = use_position
        self.pos = FourierPosition(n_freqs) if use_position else None
        in_dim = embed_dim + (self.pos.out_dim if use_position else 0)
        self.norm = nn.LayerNorm(in_dim)
        self.mlp = nn.Sequential(nn.Linear(in_dim, llm_dim), nn.GELU(), nn.Linear(llm_dim, llm_dim))
        self.scale = nn.Parameter(torch.tensor(float(output_scale)))

    def forward(self, cell_emb: torch.Tensor, rel_pos: torch.Tensor | None = None) -> torch.Tensor:
        """cell_emb: (B, N, D), rel_pos: (B, N, 2) -> (B, N, llm_dim)"""
        x = cell_emb
        if self.use_position:
            x = torch.cat([x, self.pos(rel_pos)], dim=-1)
        return self.mlp(self.norm(x)) * self.scale
