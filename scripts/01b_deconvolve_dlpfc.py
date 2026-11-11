"""DLPFC Visium 细胞类型反卷积(见 src/spatial_reason/data/deconvolve.py)。

先下载 spatialDLPFC 单核参考(约 4 GB,spatialLIBD::fetch_data("spatialDLPFC_snRNAseq") 的同一文件):

    mkdir -p data/raw/dlpfc_snrna && curl -L -o data/raw/dlpfc_snrna/sce_DLPFC_annotated.zip \
        "https://www.dropbox.com/s/5919zt00vm1ht8e/sce_DLPFC_annotated.zip?dl=1"
    unzip -q data/raw/dlpfc_snrna/sce_DLPFC_annotated.zip -d data/raw/dlpfc_snrna -x '__MACOSX/*'
    python scripts/01b_deconvolve_dlpfc.py

输出 data/processed/dlpfc_deconv/{sample}.csv,并把比例写回 data/processed/dlpfc/{sample}.h5ad
(obsm['cell_type_props'],obs['cell_type'] 为主导类型)。最后按真值层打印平均比例,用于检查。
"""

import argparse
import json
import subprocess
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd

from spatial_reason.data.datasets import PROCESSED_ROOT
from spatial_reason.data.prepare import SOURCES, standardize
from spatial_reason.data.deconvolve import PROPS_DIR, REF_ROOT, attach_props, build_signatures, deconvolve, \
    platform_factors, select_markers


RAW_ROOT = "data/raw/sdmbench"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--n-markers", type=int, default=30)
    p.add_argument("--min-detect", type=float, default=0.2)
    args = p.parse_args()

    if not (REF_ROOT / "coldata.csv").exists() or not (REF_ROOT / "rowdata.csv").exists():
        subprocess.run(["Rscript", "scripts/r/export_dlpfc_snrna.R", str(REF_ROOT)], check=True)
    cache = REF_ROOT / "signatures.npz"
    if cache.exists():
        sig = dict(np.load(cache))
    else:
        sig = build_signatures()
        np.savez(cache, **sig)
    print("[ref] cells per type:", dict(zip(sig["types"].tolist(), sig["n"].astype(int).tolist())))

    samples = sorted(x.removesuffix(".h5ad") for x in SOURCES["dlpfc"].samples)
    gene_ids = set(ad.read_h5ad(Path(RAW_ROOT) / "dlpfc" / f"{samples[0]}.h5ad", backed="r").var["gene_ids"].astype(str))
    usable = np.isin(sig["gene_id"], list(gene_ids))
    markers = select_markers(sig, usable, args.n_markers, args.min_detect)
    print(f"[markers] {len(markers)} genes ({usable.sum()} shared with Visium)")

    # 从原始文件重新标准化,不依赖 data/processed 里已有的(可能是上次写了一半的)h5ad
    spec = SOURCES["dlpfc"]
    adatas = {s: standardize(ad.read_h5ad(Path(RAW_ROOT) / "dlpfc" / f"{s}.h5ad"), spec, s,
                             spec.samples[f"{s}.h5ad"][1]) for s in samples}
    factors = platform_factors(list(adatas.values()), sig, markers)
    PROPS_DIR.mkdir(parents=True, exist_ok=True)
    (PROPS_DIR / "markers.json").write_text(json.dumps(
        {"n_markers": args.n_markers, "min_detect": args.min_detect,
         "genes": sig["gene_name"][markers].tolist(),
         "platform_factors": dict(zip(sig["gene_name"][markers].tolist(), factors.round(4).tolist()))}, indent=1))
    by_layer = []
    for s, a in adatas.items():
        P = deconvolve(a, sig, markers, factors)
        P.to_csv(PROPS_DIR / f"{s}.csv")
        attach_props(a, s)
        a.write_h5ad(Path(PROCESSED_ROOT) / "dlpfc" / f"{s}.h5ad", compression="gzip")
        by_layer.append(P.assign(domain=a.obs["domain"].astype(object).values))
        print(f"[ok] {s}: {len(P)} spots, dominant types {a.obs['cell_type'].value_counts().head(4).to_dict()}")

    tab = pd.concat(by_layer).dropna(subset=["domain"]).groupby("domain").mean()
    pd.set_option("display.width", 250)
    print("\n平均比例(行:真值层,列:细胞类型)\n", (100 * tab).round(0).astype(int).to_string())
    tab.to_csv(PROPS_DIR / "mean_props_by_layer.csv")


if __name__ == "__main__":
    main()
