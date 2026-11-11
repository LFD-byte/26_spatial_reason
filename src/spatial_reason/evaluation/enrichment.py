"""富集判定(阈值 / 统计检验),供证据奖励、轨迹过滤、忠实性评测共用。

一个邻域(中心 + k 个邻居)相对整张切片:
- 细胞类型富集:邻域占比 >= min_frac,且(占比 / 切片占比 >= min_ratio 或 占比 >= dominant_frac);
- 基因富集:效应量 (邻域均值 - 切片均值) / 切片标准差 >= min_effect,且邻域检出比例 >= min_detect
  (与 descriptions.DescriptionBuilder.enriched_genes 同口径)。
阈值是方案第 11 节的待定项,集中放在 EnrichmentCriteria 里。
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from spatial_reason.data.descriptions import DescriptionBuilder


@dataclass(frozen=True)
class EnrichmentCriteria:
    min_frac: float = 0.10
    min_ratio: float = 1.5
    dominant_frac: float = 0.50
    min_effect: float = 0.5
    min_detect: float = 0.3


@dataclass
class NeighborhoodStats:
    center_type: str | None
    composition: dict[str, float]  # 邻域占比(>0 的类型)
    slice_composition: dict[str, float]  # 同一类型在整张切片的占比
    gene_effect: dict[str, float]  # 满足检出比例的基因 -> 效应量(只存前 top_genes 个)
    enriched_types: list[str]
    enriched_genes: list[str]
    criteria: dict

    def to_dict(self) -> dict:
        return asdict(self)


class EnrichmentCalculator:
    def __init__(self, desc: DescriptionBuilder, criteria: EnrichmentCriteria | None = None):
        self.desc, self.c = desc, criteria or EnrichmentCriteria()
        if desc.has_cell_type:
            counts = np.bincount(desc.type_codes, minlength=len(desc.type_names))
            self.slice_frac = counts / counts.sum()

    def stats(self, idx: np.ndarray, top_genes: int = 10) -> NeighborhoodStats:
        d, c = self.desc, self.c
        comp, slice_comp, enriched_types, center = {}, {}, [], None
        if d.has_cell_type:
            frac = np.bincount(d.type_codes[idx], minlength=len(d.type_names)) / len(idx)
            center = str(d.type_names[d.type_codes[idx[0]]])
            for j in np.argsort(-frac):
                if frac[j] <= 0:
                    break
                name = str(d.type_names[j])
                comp[name] = round(float(frac[j]), 3)
                slice_comp[name] = round(float(self.slice_frac[j]), 3)
                ratio = frac[j] / max(self.slice_frac[j], 1e-9)
                if frac[j] >= c.min_frac and (ratio >= c.min_ratio or frac[j] >= c.dominant_frac):
                    enriched_types.append(name)

        sub = d.X[idx]
        m = np.asarray(sub.mean(0)).ravel()
        detect = np.asarray((sub > 0).mean(0)).ravel()
        effect = (m - d.mu) / d.sd
        ok = (detect >= c.min_detect) & (m > d.mu)
        effect = np.where(ok, effect, -np.inf)
        order = np.argsort(-effect)
        gene_effect = {str(d.genes[j]): round(float(effect[j]), 2) for j in order[:top_genes]
                       if effect[j] >= c.min_effect}
        enriched_genes = [str(d.genes[j]) for j in order if effect[j] >= c.min_effect]
        return NeighborhoodStats(center_type=center, composition=comp, slice_composition=slice_comp,
                                 gene_effect=gene_effect, enriched_types=enriched_types,
                                 enriched_genes=enriched_genes, criteria=asdict(c))
