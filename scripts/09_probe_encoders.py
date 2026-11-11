"""编码器对比:同一批邻域、同一 niche 标签,用 logistic 回归线性探针比较不同细胞表示。

特征 = 中心细胞表示 ⊕ 邻域(33 个细胞)平均表示。邻域与 stage 1 教师请求相同(train / val 切片划分一致)。

    # Nicheformer 只对探针用到的细胞提取(逐细胞模型,与空间上下文无关,可以只算子集);两卡各跑一半
    CUDA_VISIBLE_DEVICES=0 python scripts/09_probe_encoders.py nicheformer --part 0 --nparts 2 &
    CUDA_VISIBLE_DEVICES=1 python scripts/09_probe_encoders.py nicheformer --part 1 --nparts 2
    python scripts/09_probe_encoders.py probe

表示:
- novae        Novae 零样本嵌入(64 维,带空间上下文)
- novae_ft     在训练切片上自监督微调后的 Novae(scripts/02b_finetune_novae.py)
- novae_scratch 不用预训练权重、只在训练切片上从头训练的 Novae(同上,--from-scratch)
- pca          normalize_total + log1p 表达的 PCA(64 维,在训练切片上拟合)
- novae+pca    两者拼接
- nicheformer  Nicheformer 嵌入(512 维;只在 --nf-datasets 上)
- oracle_comp  人工注释细胞类型的组成比例 + 中心类型 one-hot(纯文本教师看到的就是这个,参照上限)
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from spatial_reason.data.datasets import DATASET_INFO, load_sample
from spatial_reason.data.neighborhoods import build_neighborhoods
from spatial_reason.encoders.base import load_embeddings

TRACES = Path("data/traces/qwen3_32b")
OUT = Path("outputs/probes")
NF_CACHE = Path("data/embeddings/nicheformer_probe")
K = 32


def load_requests(datasets, max_train: dict[str, int], seed: int = 0) -> dict:
    """每个数据集的 train / val 邻域;train 按标签分层下采样到 max_train。"""
    rng = np.random.default_rng(seed)
    out = {}
    for ds in datasets:
        sp = {}
        for split in ("train", "val"):
            rs = [r for r in map(json.loads, open(TRACES / f"requests_{split}.jsonl")) if r["dataset"] == ds]
            for r in rs:
                r.pop("messages", None)
            cap = max_train.get(ds) if split == "train" else None
            if cap and len(rs) > cap:
                by = defaultdict(list)
                for r in rs:
                    by[r["label"]].append(r)
                per = cap // len(by)
                rs = [r for g in by.values() for r in rng.permutation(g)[:per].tolist()]
            sp[split] = rs
        out[ds] = sp
    return out


class Slice:
    def __init__(self, ds: str, sample: str):
        self.adata = load_sample(ds, sample)
        self.nb = build_neighborhoods(self.adata.obsm["spatial"], K)
        self.reps: dict[str, np.ndarray] = {}

    def idx(self, center: int) -> np.ndarray:
        return self.nb.index[center]


def log_expr(adata) -> np.ndarray:
    import scanpy as sc

    a = adata.copy()
    import scipy.sparse as sp

    X = a.X
    vals = X.data if sp.issparse(X) else np.asarray(X).ravel()
    is_int = np.allclose(vals[:100000] % 1, 0)
    if is_int:
        sc.pp.normalize_total(a, target_sum=1e4)
    sc.pp.log1p(a)  # MERFISH X 已按体积归一化(非整数):只做 log1p
    return a.X.toarray() if hasattr(a.X, "toarray") else np.asarray(a.X)


def fit_pca(slices: dict, train_samples: set, n: int = 64, seed: int = 0):
    from sklearn.decomposition import PCA

    rng = np.random.default_rng(seed)
    logs = {s: log_expr(sl.adata) for s, sl in slices.items()}
    fit = np.concatenate([L[rng.choice(len(L), min(len(L), 50000), replace=False)]
                          for s, L in logs.items() if s in train_samples])
    pca = PCA(n_components=min(n, fit.shape[1] - 1), random_state=seed).fit(fit)
    for s, sl in slices.items():
        sl.reps["pca"] = pca.transform(logs[s]).astype(np.float32)


def oracle_comp(sl: Slice, types: list[str]) -> None:
    ct = sl.adata.obs["cell_type"].astype(str).values
    sl.reps["_ct_onehot"] = (ct[:, None] == np.array(types)[None, :]).astype(np.float32)


def features(sl: Slice, rep: str, center: int) -> np.ndarray:
    idx = sl.idx(center)
    if rep == "oracle_comp":
        oh = sl.reps["_ct_onehot"]
        return np.concatenate([oh[idx[0]], oh[idx].mean(0)])
    if rep == "novae+pca":
        return np.concatenate([features(sl, "novae", center), features(sl, "pca", center)])
    z = sl.reps[rep]  # nicheformer 只算了探针用到的细胞,其余行为 nan(features 只会取到算过的行)
    return np.concatenate([z[idx[0]], z[idx].mean(0)])


def probe_cells(reqs: dict, slices: dict) -> dict[str, np.ndarray]:
    cells = defaultdict(set)
    for split in reqs.values():
        for r in split:
            cells[r["sample"]].update(slices[r["sample"]].idx(r["center"]).tolist())
    return {s: np.array(sorted(c)) for s, c in cells.items()}


def cmd_nicheformer(a) -> None:
    from spatial_reason.encoders.nicheformer import NicheformerEncoder

    NF_CACHE.mkdir(parents=True, exist_ok=True)
    enc = None
    reqs = load_requests(a.nf_datasets, {d: a.max_train for d in a.nf_datasets})
    for ds in a.nf_datasets:
        samples = sorted({r["sample"] for sp in reqs[ds].values() for r in sp})
        slices = {s: Slice(ds, s) for s in samples}
        cells = probe_cells(reqs[ds], slices)
        jobs = [(s, c) for s in samples for c in np.array_split(cells[s], a.nparts)[a.part:a.part + 1]]
        for s, c in jobs:
            f = NF_CACHE / f"{ds}__{s}__part{a.part}of{a.nparts}.npz"
            if f.exists():
                continue
            enc = enc or NicheformerEncoder()
            info = DATASET_INFO[ds]
            z, _ = enc.embed(slices[s].adata[c].copy(), species=info["species"], platform=info["platform"].lower(),
                             batch_size=64)
            np.savez(f, cells=c, z=z.astype(np.float32))
            print(f"[nf] {ds}/{s} part {a.part}: {len(c)} cells", flush=True)


def load_nf(ds: str, sl: Slice, sample: str) -> bool:
    parts = sorted(NF_CACHE.glob(f"{ds}__{sample}__part*.npz"))
    if not parts:
        return False
    z = None
    for p in parts:
        d = np.load(p)
        if z is None:
            z = np.full((sl.adata.n_obs, d["z"].shape[1]), np.nan, dtype=np.float32)
        z[d["cells"]] = d["z"]
    sl.reps["nicheformer"] = z
    return True


def cmd_probe(a) -> None:
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import f1_score
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    OUT.mkdir(parents=True, exist_ok=True)
    caps = {"cosmx_nsclc": a.max_train}
    reqs = load_requests(a.datasets, caps)
    results = {}
    for ds in a.datasets:
        samples = sorted({r["sample"] for sp in reqs[ds].values() for r in sp})
        slices = {s: Slice(ds, s) for s in samples}
        types = sorted({t for sl in slices.values() for t in sl.adata.obs["cell_type"].astype(str)})
        for s, sl in slices.items():
            sl.reps["novae"] = load_embeddings("novae", ds, s, adata=sl.adata)
            for extra in ("novae_ft", "novae_scratch"):
                try:
                    sl.reps[extra] = load_embeddings(extra, ds, s, adata=sl.adata)
                except FileNotFoundError:
                    pass
            oracle_comp(sl, types)
        fit_pca(slices, {r["sample"] for r in reqs[ds]["train"]})
        reps = ["novae", "pca", "novae+pca", "oracle_comp"]
        for i, extra in enumerate(("novae_ft", "novae_scratch")):
            if all(extra in sl.reps for sl in slices.values()):
                reps.insert(1 + i, extra)
        if all(load_nf(ds, slices[s], s) for s in samples):
            reps.insert(3, "nicheformer")
        ytr = [r["label"] for r in reqs[ds]["train"]]
        yva = [r["label"] for r in reqs[ds]["val"]]
        for rep in reps:
            Xtr = np.stack([features(slices[r["sample"]], rep, r["center"]) for r in reqs[ds]["train"]])
            Xva = np.stack([features(slices[r["sample"]], rep, r["center"]) for r in reqs[ds]["val"]])
            assert np.isfinite(Xtr).all() and np.isfinite(Xva).all(), f"{ds}/{rep}: 有细胞缺少表示"
            clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=4000, C=a.C, class_weight="balanced"))
            p = clf.fit(Xtr, ytr).predict(Xva)
            res = {"acc": float(np.mean(p == np.array(yva))), "macro_f1": float(f1_score(yva, p, average="macro")),
                   "dim": Xtr.shape[1], "n_train": len(ytr), "n_val": len(yva)}
            results[f"{ds}/{rep}"] = res
            print(f"{ds:22s} {rep:12s} acc={res['acc']:.3f} f1={res['macro_f1']:.3f} "
                  f"(dim {res['dim']}, train {len(ytr)}, val {len(yva)})", flush=True)
    (OUT / "encoder_probe.json").write_text(json.dumps(results, indent=2))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("cmd", choices=["nicheformer", "probe"])
    p.add_argument("--datasets", nargs="+", default=["cosmx_nsclc", "merfish_hypothalamus", "starmap"])
    p.add_argument("--nf-datasets", nargs="+", default=["cosmx_nsclc"])
    p.add_argument("--max-train", type=int, default=4000, help="CosMx 训练邻域分层下采样数(控制 Nicheformer 提取量)")
    p.add_argument("--part", type=int, default=0)
    p.add_argument("--nparts", type=int, default=1)
    p.add_argument("--C", type=float, default=0.5)
    a = p.parse_args()
    (cmd_nicheformer if a.cmd == "nicheformer" else cmd_probe)(a)


if __name__ == "__main__":
    main()
