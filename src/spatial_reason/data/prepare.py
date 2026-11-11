"""把各来源的原始 h5ad 转成统一格式,写到 data/processed/{dataset}/{sample}.h5ad。

统一格式:
- X:原文件中的表达矩阵(不做归一化,归一化交给编码器);var_names 为基因 symbol
- obs['cell_type']:细胞类型(字符串);Visium 为 spot 的主导类型,比例在 obsm['cell_type_props'](反卷积)
- obs['domain']:空间域真值(字符串);缺失为 NaN
- obs['sample'] / obs['donor'] / obs['platform']
- obsm['spatial']:二维坐标,float64
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd


@dataclass(frozen=True)
class SourceSpec:
    platform: str
    cell_type_key: str | None
    domain_key: str
    # 原文件名 -> (sample, donor)
    samples: dict[str, tuple[str, str]]


SOURCES: dict[str, SourceSpec] = {
    "dlpfc": SourceSpec(
        platform="visium",
        cell_type_key=None,
        domain_key="Region",
        samples={
            **{f"{s}.h5ad": (s, "Br5292") for s in ["151507", "151508", "151509", "151510"]},
            **{f"{s}.h5ad": (s, "Br5595") for s in ["151669", "151670", "151671", "151672"]},
            **{f"{s}.h5ad": (s, "Br8100") for s in ["151673", "151674", "151675", "151676"]},
        },
    ),
    "merfish_hypothalamus": SourceSpec(
        platform="merfish",
        cell_type_key="cell_class",
        domain_key="domain",
        # Moffitt 2018 Animal 1,切片按 Bregma 命名
        samples={f"MERFISH_{b}.h5ad": (f"bregma_-{b}", "animal1")
                 for b in ["0.04", "0.09", "0.14", "0.19", "0.24"]},
    ),
    "starmap": SourceSpec(
        platform="starmap",
        cell_type_key="ct",
        domain_key="region",
        samples={
            "20180417_BZ5_control.h5ad": ("BZ5", "BZ5"),
            "20180419_BZ9_control.h5ad": ("BZ9", "BZ9"),
            "20180424_BZ14_control.h5ad": ("BZ14", "BZ14"),
        },
    ),
}


def _as_str(s: pd.Series) -> pd.Series:
    out = s.astype(object).where(s.notna(), None)
    out = out.map(lambda v: None if v is None else str(v))
    return out.astype("category")


def standardize(adata: ad.AnnData, spec: SourceSpec, sample: str, donor: str) -> ad.AnnData:
    obs = pd.DataFrame(index=adata.obs_names.astype(str))
    obs["sample"] = sample
    obs["donor"] = donor
    obs["platform"] = spec.platform
    obs["domain"] = _as_str(adata.obs[spec.domain_key]).values
    if spec.cell_type_key is not None:
        obs["cell_type"] = _as_str(adata.obs[spec.cell_type_key]).values
    else:
        obs["cell_type"] = pd.Categorical([None] * adata.n_obs)
    # Visium:保留阵列坐标供后续使用
    for k in ("array_row", "array_col"):
        if k in adata.obs:
            obs[k] = adata.obs[k].values

    var = pd.DataFrame(index=adata.var_names.astype(str))
    if "gene_ids" in adata.var:
        var["gene_ids"] = adata.var["gene_ids"].values

    out = ad.AnnData(X=adata.X, obs=obs, var=var)
    out.var_names_make_unique()
    out.obsm["spatial"] = np.asarray(adata.obsm["spatial"], dtype=np.float64)[:, :2]
    out.obs_names = [f"{sample}_{i}" for i in out.obs_names]
    return out


def prepare_dataset(
    name: str,
    raw_root: Path | str = "data/raw/sdmbench",
    out_root: Path | str = "data/processed",
) -> pd.DataFrame:
    """处理一个数据集的所有切片,返回每张切片的汇总表(并写到 summary.csv)。"""
    spec = SOURCES[name]
    raw_dir, out_dir = Path(raw_root) / name, Path(out_root) / name
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for fname, (sample, donor) in spec.samples.items():
        a = standardize(ad.read_h5ad(raw_dir / fname), spec, sample, donor)
        if name == "dlpfc":  # 反卷积比例(scripts/01b_deconvolve_dlpfc.py),重跑预处理时保留
            from spatial_reason.data.deconvolve import attach_props

            attach_props(a, sample)
        a.write_h5ad(out_dir / f"{sample}.h5ad", compression="gzip")
        rows.append({
            "dataset": name,
            "sample": sample,
            "donor": donor,
            "n_obs": a.n_obs,
            "n_genes": a.n_vars,
            "n_domain": a.obs["domain"].nunique(),
            "domain_missing": int(a.obs["domain"].isna().sum()),
            "n_cell_type": a.obs["cell_type"].nunique(),
            "domains": "|".join(sorted(a.obs["domain"].dropna().unique())),
        })
        print(f"[ok] {name}/{sample}: {a.n_obs} obs x {a.n_vars} genes")

    summary = pd.DataFrame(rows)
    summary.to_csv(out_dir / "summary.csv", index=False)
    return summary


# ---------------------------------------------------------------- CosMx NSCLC

COSMX_PX_UM = 0.18  # CosMx 像素尺寸(µm/px)
_COSMX_KEEP = ("_exprMat_file.csv", "_metadata_file.csv", "_fov_positions_file.csv")


def _extract_cosmx_flat(raw_dir: Path, sample: str) -> Path:
    """只解出表达矩阵、元数据和 FOV 位置三个 csv(跳过几 GB 的转录本文件和图像)。"""
    import tarfile

    flat_dir = raw_dir / "flat" / sample
    if all(any(flat_dir.rglob(f"*{s}")) for s in _COSMX_KEEP):
        return flat_dir
    with tarfile.open(raw_dir / f"{sample} SMI Flat data.tar.gz") as tf:
        members = [m for m in tf if m.name.endswith(_COSMX_KEEP)]
        tf.extractall(raw_dir / "flat", members=members, filter="data")
    return flat_dir


def prepare_cosmx(
    raw_root: Path | str = "data/raw/cosmx_nsclc",
    out_root: Path | str = "data/processed",
    samples: list[str] | None = None,
) -> pd.DataFrame:
    """CosMx NSCLC:Flat data 的计数 + Giotto 对象导出的 cell_type / niche 注释。

    需要先运行 scripts/r/export_cosmx_giotto.R 生成 cell_annotations.csv。
    只保留有注释的细胞;`domain` 取 niche;坐标为全局像素 × 0.18,单位 µm。
    """
    import scipy.sparse as sp

    from spatial_reason.data.download import COSMX_SAMPLES

    raw_dir, out_dir = Path(raw_root), Path(out_root) / "cosmx_nsclc"
    out_dir.mkdir(parents=True, exist_ok=True)

    ann = pd.read_csv(raw_dir / "cell_annotations.csv")
    parts = ann["cell_ID"].str.split("_", expand=True)  # c_{slide}_{fov}_{cell}
    ann["fov"] = parts[2].astype(int)
    ann["cell"] = parts[3].astype(int)

    rows = []
    for sample in samples or COSMX_SAMPLES:
        flat = _extract_cosmx_flat(raw_dir, sample)
        expr = pd.read_csv(next(flat.rglob("*_exprMat_file.csv")))
        meta = pd.read_csv(next(flat.rglob("*_metadata_file.csv")))
        expr = expr[expr["cell_ID"] != 0]  # cell_ID 0 = 未分配到细胞的背景转录本

        a_s = ann[ann["Run_Tissue_name"] == sample]
        df = (expr.merge(meta[["fov", "cell_ID", "CenterX_global_px", "CenterY_global_px"]],
                         on=["fov", "cell_ID"])
                  .merge(a_s.drop(columns=["fov", "cell_ID"]).rename(columns={"cell": "cell_ID"})
                            .assign(fov=a_s["fov"].values),
                         on=["fov", "cell_ID"]))

        genes = [c for c in expr.columns[2:] if not c.startswith("NegPrb")]
        neg = [c for c in expr.columns[2:] if c.startswith("NegPrb")]

        obs = pd.DataFrame(index=[f"{sample}_{f}_{c}" for f, c in zip(df["fov"], df["cell_ID"])])
        obs["sample"] = sample
        obs["donor"] = df["patient"].values
        obs["platform"] = "cosmx"
        obs["domain"] = pd.Categorical(df["niche"].values)
        obs["cell_type"] = pd.Categorical(df["cell_type"].values)
        obs["fov"] = df["fov"].values
        obs["remove_flagged_cells"] = df["remove_flagged_cells"].values
        obs["prop_tumor_in_100_neighbors"] = df["prop_tumor_in_100_neighbors"].values
        obs["neg_probe_counts"] = df[neg].sum(1).values

        a = ad.AnnData(
            X=sp.csr_matrix(df[genes].to_numpy(dtype=np.float32)),
            obs=obs,
            var=pd.DataFrame(index=genes),
        )
        a.obsm["spatial"] = df[["CenterX_global_px", "CenterY_global_px"]].to_numpy(np.float64) * COSMX_PX_UM
        a.write_h5ad(out_dir / f"{sample}.h5ad", compression="gzip")

        rows.append({
            "dataset": "cosmx_nsclc", "sample": sample, "donor": obs["donor"].iloc[0],
            "n_obs": a.n_obs, "n_annotated": len(a_s), "n_genes": a.n_vars,
            "n_domain": obs["domain"].nunique(), "domain_missing": int(obs["domain"].isna().sum()),
            "n_cell_type": obs["cell_type"].nunique(),
            "domains": "|".join(sorted(obs["domain"].dropna().unique())),
        })
        print(f"[ok] cosmx_nsclc/{sample}: {a.n_obs} cells (annotated {len(a_s)}) x {a.n_vars} genes")

    summary = pd.DataFrame(rows)
    summary.to_csv(out_dir / "summary.csv", index=False)
    return summary
