"""无监督空间域划分对比:不用任何标签,每张切片聚成 k 个域(k = 该切片真实域数,与 LLMiniST 相同),与注释比 ARI / NMI。

这是 Novae 设计上要做的事(assign_domains),用来补充线性探针:探针偏向保留原始表达信息的表示(PCA),
这里测的是"不给标签时能不能自己分出组织结构"。

方法:
- novae*_native   Novae 自带的层次原型划分(compute_representations + assign_domains(n_domains=k))
- novae*_kmeans   同一 Novae 表示上做 k-means
- pca_cell        每个细胞自己的 PCA 表示做 k-means(无空间信息,下界)
- pca_nbr         33 个细胞(中心 + 32 近邻)的 PCA 平均后做 k-means(设计文档里的朴素对照)
- oracle_nbr      33 个细胞的人工注释细胞类型组成做 k-means(参照)

    CUDA_VISIBLE_DEVICES= python scripts/10_domain_clustering.py
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np

from spatial_reason.data.datasets import domain_name, list_samples, load_sample
from spatial_reason.data.neighborhoods import build_neighborhoods
from spatial_reason.encoders.base import load_embeddings
from spatial_reason.encoders.novae import DATASET_SETTINGS, FT_ROOT, SCRATCH_ROOT, prepare_for_novae
from spatial_reason.utils.config import load_config

OUT = Path("outputs/domains")
K = 32


def kmeans(X: np.ndarray, k: int, seed: int = 0) -> np.ndarray:
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler

    return KMeans(k, n_init=10, random_state=seed).fit_predict(StandardScaler().fit_transform(X))


def native_domains(model, na, k: int) -> np.ndarray:
    """Novae 层次原型树上划出 k 个域;切片里只出现部分叶子时可能没有恰好 k 个的层级,取出现域数最接近 k 的层级。"""
    try:
        return na.obs[model.assign_domains(na, n_domains=k)].astype(str).values
    except ValueError:
        best = None
        for level in range(k, 4 * k + 1):
            d = na.obs[model.assign_domains(na, level=level)].astype(str).values
            n = len(set(d) - {"nan"})
            if best is None or abs(n - k) < best[0]:
                best = (abs(n - k), d)
            if n >= k:
                break
        return best[1]


def novae_models(ds: str) -> dict:
    """可用的 Novae 变体:name -> (模型路径, zero_shot)。"""
    s = DATASET_SETTINGS[ds]
    out = {"novae": (f"data/models/{s['model']}", True)}
    for name, root in (("novae_ft", FT_ROOT), ("novae_scratch", SCRATCH_ROOT)):
        if (Path(root) / ds).exists():
            out[name] = (f"{root}/{ds}", False)
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--datasets", nargs="+", default=["starmap", "merfish_hypothalamus", "cosmx_nsclc"])
    p.add_argument("--config", default="configs/train/stage0_align.yaml", help="取 val_samples,单独汇总验证切片")
    p.add_argument("--accelerator", default="cpu")
    p.add_argument("--num-workers", type=int, default=8)
    a = p.parse_args()

    import novae
    from sklearn.metrics import adjusted_rand_score as ari
    from sklearn.metrics import normalized_mutual_info_score as nmi

    cfg = load_config(a.config)
    OUT.mkdir(parents=True, exist_ok=True)
    out_path = OUT / "domain_clustering.json"
    results = json.loads(out_path.read_text()) if out_path.exists() else {}
    for ds in a.datasets:
        val = set(cfg["data"]["datasets"][ds]["val_samples"])
        models = {n: novae.Novae.from_pretrained(path) for n, (path, _) in novae_models(ds).items()}
        zero_shot = {n: z for n, (_, z) in novae_models(ds).items()}
        for s in list_samples(ds):
            t0 = time.time()
            adata = load_sample(ds, s)
            y = np.array([domain_name(ds, v) for v in adata.obs["domain"].astype(object)], dtype=object)
            m = np.array([v is not None for v in y])
            k = len(set(y[m]))
            nb = build_neighborhoods(adata.obsm["spatial"], K).index  # (n, 33),第 0 列为中心
            preds = {}
            for name, model in models.items():
                na, _ = prepare_for_novae(adata, ds)
                model.compute_representations(na, zero_shot=zero_shot[name], accelerator=a.accelerator,
                                              num_workers=a.num_workers)
                preds[f"{name}_native"] = native_domains(model, na, k)
                preds[f"{name}_kmeans"] = kmeans(np.asarray(na.obsm["novae_latent"]), k)
            z = load_embeddings("pca", ds, s, adata=adata)
            preds["pca_cell"] = kmeans(z, k)
            preds["pca_nbr"] = kmeans(z[nb].mean(1), k)
            ct = adata.obs["cell_type"].astype(str).values
            types = np.unique(ct)
            onehot = (ct[:, None] == types[None, :]).astype(np.float32)
            preds["oracle_nbr"] = kmeans(onehot[nb].mean(1), k)
            for meth, pr in preds.items():
                pr = np.asarray(pr)
                ok = m & (pr != "nan")
                results[f"{ds}/{s}/{meth}"] = {
                    "dataset": ds, "sample": s, "method": meth, "split": "val" if s in val else "train",
                    "k": k, "n": int(ok.sum()),
                    "ari": float(ari(y[ok].astype(str), pr[ok])), "nmi": float(nmi(y[ok].astype(str), pr[ok]))}
            line = "  ".join(f"{meth}={results[f'{ds}/{s}/{meth}']['ari']:.2f}" for meth in preds)
            print(f"[{ds}/{s}] k={k} n={adata.n_obs} ({time.time() - t0:.0f}s) ARI {line}", flush=True)
            out_path.write_text(json.dumps(results, indent=2))
    summarize(results)


def summarize(results: dict) -> None:
    from collections import defaultdict

    agg = defaultdict(lambda: defaultdict(list))
    for r in results.values():
        for split in ("all", r["split"]):
            agg[(r["dataset"], r["method"])][split].append((r["ari"], r["nmi"]))
    print(f"\n{'dataset':22s} {'method':22s} {'ARI(all)':>9s} {'NMI(all)':>9s} {'ARI(val)':>9s} {'NMI(val)':>9s}")
    for (ds, meth), d in sorted(agg.items()):
        al, va = np.array(d["all"]), np.array(d.get("val", [[np.nan, np.nan]]))
        print(f"{ds:22s} {meth:22s} {al[:, 0].mean():9.3f} {al[:, 1].mean():9.3f} "
              f"{va[:, 0].mean():9.3f} {va[:, 1].mean():9.3f}")


if __name__ == "__main__":
    main()
