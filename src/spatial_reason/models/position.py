"""邻居相对坐标编码:固定频率的 Fourier 特征。

输入是除以中位最近邻距离后的相对坐标 (dx, dy);中心细胞为 (0, 0)。
特征 = [sin(2π f dx), cos(2π f dx), sin(2π f dy), cos(2π f dy)] × n_freqs,外加 log(1 + 距离)。
频率按几何级数覆盖 1/64 到 1 个细胞间距的周期,k=32 时邻域半径约 3–4 个间距。
"""

from __future__ import annotations

import math

import torch
from torch import nn


class FourierPosition(nn.Module):
    def __init__(self, n_freqs: int = 8, min_period: float = 1.0, max_period: float = 64.0):
        super().__init__()
        periods = torch.logspace(math.log10(min_period), math.log10(max_period), n_freqs)
        self.register_buffer("freqs", 2 * math.pi / periods, persistent=False)
        self.out_dim = 4 * n_freqs + 1

    def forward(self, rel_pos: torch.Tensor) -> torch.Tensor:
        """rel_pos: (..., 2) -> (..., out_dim)"""
        ang = rel_pos.unsqueeze(-1) * self.freqs  # (..., 2, F)
        feats = torch.cat([ang.sin(), ang.cos()], dim=-1).flatten(-2)  # (..., 4F)
        dist = torch.log1p(rel_pos.norm(dim=-1, keepdim=True))
        return torch.cat([feats, dist], dim=-1)
