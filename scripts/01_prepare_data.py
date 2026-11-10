"""下载并标准化数据集。

    python scripts/01_prepare_data.py                    # 全部
    python scripts/01_prepare_data.py --datasets starmap --skip-download
    python scripts/01_prepare_data.py --datasets cosmx_nsclc   # 需要 Rscript
"""

import argparse
import subprocess
from pathlib import Path

from spatial_reason.data.download import SDMBENCH_ARTICLES, download_cosmx, download_dataset
from spatial_reason.data.prepare import prepare_cosmx, prepare_dataset

COSMX_RAW = "data/raw/cosmx_nsclc"


def run_cosmx(skip_download: bool, out_root: str) -> None:
    raw = Path(COSMX_RAW)
    if not skip_download:
        download_cosmx(raw)
    rdata = raw / "giotto" / "Processed Data Giotto Object" / "SMI_Giotto_Object.RData"
    if not rdata.exists():
        (raw / "giotto").mkdir(exist_ok=True)
        subprocess.run(["tar", "-xzf", str(raw / "All SMI Giotto object.tar.gz"),
                        "-C", str(raw / "giotto")], check=True)
    if not (raw / "cell_annotations.csv").exists():
        subprocess.run(["Rscript", "scripts/r/export_cosmx_giotto.R", str(raw)], check=True)
    print(prepare_cosmx(raw, out_root).drop(columns="domains"))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--datasets", nargs="+", default=[*SDMBENCH_ARTICLES, "cosmx_nsclc"])
    p.add_argument("--raw-root", default="data/raw/sdmbench")
    p.add_argument("--out-root", default="data/processed")
    p.add_argument("--skip-download", action="store_true")
    args = p.parse_args()

    for name in args.datasets:
        if name == "cosmx_nsclc":
            run_cosmx(args.skip_download, args.out_root)
            continue
        if not args.skip_download:
            download_dataset(name, args.raw_root)
        print(prepare_dataset(name, args.raw_root, args.out_root).drop(columns="domains"))


if __name__ == "__main__":
    main()
