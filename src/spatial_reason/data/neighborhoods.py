"""目标细胞 + k 个邻居的邻域构造、相对坐标;区域级任务的微区域池化。

相对坐标除以该切片的中位最近邻距离,单位变成"典型细胞间距",跨平台可比
(CosMx/MERFISH 为 µm,STARmap/Visium 为像素)。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.neighbors import NearestNeighbors


@dataclass
class Neighborhoods:
    index: np.ndarray  # (n, k+1) int64,第 0 列是中心细胞自身,其余按距离升序
    rel_pos: np.ndarray  # (n, k+1, 2) float32,相对中心的坐标 / spacing
    spacing: float  # 中位最近邻距离(原坐标单位)

    @property
    def k(self) -> int:
        return self.index.shape[1] - 1


def build_neighborhoods(xy: np.ndarray, k: int = 32) -> Neighborhoods:
    xy = np.asarray(xy, dtype=np.float64)
    nn = NearestNeighbors(n_neighbors=k + 1).fit(xy)
    dist, idx = nn.kneighbors(xy)
    # 坐标完全重合时自身可能不在第 0 位,强制把中心放到第 0 列
    self_first = idx[:, 0] == np.arange(len(xy))
    if not self_first.all():
        for i in np.where(~self_first)[0]:
            row = [i] + [j for j in idx[i] if j != i][:k]
            idx[i] = row
    spacing = float(np.median(dist[:, 1])) or 1.0
    rel = (xy[idx] - xy[:, None, :]) / spacing
    return Neighborhoods(index=idx.astype(np.int64), rel_pos=rel.astype(np.float32), spacing=spacing)
