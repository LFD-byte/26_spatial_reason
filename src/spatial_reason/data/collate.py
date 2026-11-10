"""Dataset / collator:把邻域嵌入、坐标与文本 prompt 组装成 batch。

一个样本 = 一个中心细胞:
- cell_emb (k+1, D):中心 + k 个邻居的编码器嵌入(第 0 个是中心);
- rel_pos (k+1, 2):相对坐标(单位:中位最近邻距离);
- prompt:chat 模板的 user 部分,含 k+1 个 `<|cell|>` 占位符;
- target:自动生成的描述(阶段 0),loss 只算在 target 上。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import Dataset

from spatial_reason.data.datasets import DATASET_INFO, load_sample
from spatial_reason.data.descriptions import DescriptionBuilder
from spatial_reason.data.neighborhoods import Neighborhoods, build_neighborhoods
from spatial_reason.encoders.base import load_embeddings
from spatial_reason.models.spatial_llm import CELL_TOKEN

STAGE0_PROMPT = (
    "Below is a cell neighborhood from a {platform} section of {tissue}: "
    "<cells>{cells}</cells>\n"
    "The first cell token is the center cell and the other {k} are its nearest neighbors, "
    "ordered by distance. Describe the cell-type composition of this neighborhood and the genes "
    "enriched in it."
)


@dataclass
class SliceData:
    dataset: str
    sample: str
    emb: np.ndarray
    nb: Neighborhoods
    desc: DescriptionBuilder
    cell_type: np.ndarray | None
    domain: np.ndarray


def load_slice(dataset: str, sample: str, encoder: str, k: int) -> SliceData:
    a = load_sample(dataset, sample)
    emb = load_embeddings(encoder, dataset, sample, adata=a)
    ct = a.obs["cell_type"].astype(object).values if a.obs["cell_type"].notna().any() else None
    return SliceData(dataset=dataset, sample=sample, emb=emb, nb=build_neighborhoods(a.obsm["spatial"], k),
                     desc=DescriptionBuilder(a, dataset), cell_type=ct,
                     domain=a.obs["domain"].astype(object).values)


class Stage0Dataset(Dataset):
    """看嵌入写描述。train 时每次随机选模板;eval 时模板固定为 i % 4,结果可复现。"""

    def __init__(self, slices: list[SliceData], centers: list[tuple[int, int]], train: bool, seed: int = 0):
        self.slices, self.centers, self.train, self.seed = slices, centers, train, seed
        self.epoch = 0

    def __len__(self) -> int:
        return len(self.centers)

    def __getitem__(self, i: int) -> dict:
        si, c = self.centers[i]
        s = self.slices[si]
        idx = s.nb.index[c]
        if self.train:
            rng = np.random.default_rng((self.seed, self.epoch, i))
            d = s.desc.describe(idx, rng=rng)
        else:
            d = s.desc.describe(idx, template=i % 4)
        info = DATASET_INFO[s.dataset]
        prompt = STAGE0_PROMPT.format(platform=info["platform"], tissue=info["tissue"],
                                      cells=CELL_TOKEN * len(idx), k=len(idx) - 1)
        return {
            "cell_emb": torch.from_numpy(s.emb[idx]),
            "rel_pos": torch.from_numpy(s.nb.rel_pos[c]),
            "prompt": prompt,
            "target": d.text,
            "meta": {"dataset": s.dataset, "sample": s.sample, "center": int(c),
                     "cell_types": d.cell_types, "genes": d.genes, "center_type": d.center},
        }


def sample_centers(slices: list[SliceData], max_per_slice: int | None, seed: int) -> list[tuple[int, int]]:
    rng = np.random.default_rng(seed)
    out = []
    for si, s in enumerate(slices):
        n = len(s.emb)
        pick = np.arange(n) if (max_per_slice is None or n <= max_per_slice) else \
            np.sort(rng.choice(n, max_per_slice, replace=False))
        out.extend((si, int(c)) for c in pick)
    return out


class Collator:
    """右侧 padding;labels 在 prompt 和 padding 处为 -100。"""

    def __init__(self, tokenizer, max_length: int = 512, with_target: bool = True):
        self.tok, self.max_length, self.with_target = tokenizer, max_length, with_target
        self.pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
        self.eot = "<|im_end|>"

    def prompt_ids(self, prompt: str) -> list[int]:
        text = self.tok.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False,
                                            add_generation_prompt=True, enable_thinking=False)
        return self.tok(text, add_special_tokens=False)["input_ids"]

    def __call__(self, batch: list[dict]) -> dict:
        ids, labels = [], []
        for b in batch:
            p = self.prompt_ids(b["prompt"])
            t = self.tok(b["target"] + self.eot, add_special_tokens=False)["input_ids"] if self.with_target else []
            seq = (p + t)[: self.max_length]
            if len(p) > self.max_length:
                raise ValueError("prompt 超过 max_length,会截掉细胞占位符")
            ids.append(seq)
            labels.append(([-100] * len(p) + t)[: self.max_length])
        if self.with_target:  # 训练:右侧 padding
            L = max(map(len, ids))
            input_ids = torch.full((len(ids), L), self.pad_id, dtype=torch.long)
            lab = torch.full((len(ids), L), -100, dtype=torch.long)
            att = torch.zeros((len(ids), L), dtype=torch.long)
            for i, (s, l) in enumerate(zip(ids, labels)):
                input_ids[i, : len(s)] = torch.tensor(s)
                lab[i, : len(l)] = torch.tensor(l)
                att[i, : len(s)] = 1
        else:  # 生成:左侧 padding
            L = max(map(len, ids))
            input_ids = torch.full((len(ids), L), self.pad_id, dtype=torch.long)
            att = torch.zeros((len(ids), L), dtype=torch.long)
            for i, s in enumerate(ids):
                input_ids[i, L - len(s):] = torch.tensor(s)
                att[i, L - len(s):] = 1
            lab = None
        out = {
            "input_ids": input_ids,
            "attention_mask": att,
            "cell_emb": torch.stack([b["cell_emb"] for b in batch]),
            "rel_pos": torch.stack([b["rel_pos"] for b in batch]),
            "meta": [b["meta"] for b in batch],
            "target": [b["target"] for b in batch],
        }
        if lab is not None:
            out["labels"] = lab
        return out
