"""下载原始数据。

当前来源:SDMBench(Yuan et al., Nat Methods 2024)在 Figshare 上的重处理 h5ad,
带空间域真值。LLMiniST 的 Figshare 数据包(10.6084/m9.figshare.33043091)尚未公开,
其 STARmap / DLPFC / MERFISH_25–29 与这里是同一批切片。
"""

from __future__ import annotations

import hashlib
import json
import urllib.parse
import urllib.request
from pathlib import Path

FIGSHARE_API = "https://api.figshare.com/v2/articles/{id}/files?page_size=100"

# dataset name -> SDMBench Figshare article id
SDMBENCH_ARTICLES = {
    "dlpfc": 22548901,  # 10x Visium, 12 slices (151507–151676), layer labels
    "merfish_hypothalamus": 22565170,  # Moffitt 2018, Animal 1, Bregma -0.04 … -0.24
    "starmap": 22565200,  # Wang 2018, mouse V1, BZ5 / BZ9 / BZ14
}


def _md5(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        while block := f.read(chunk):
            h.update(block)
    return h.hexdigest()


def list_files(article_id: int) -> list[dict]:
    with urllib.request.urlopen(FIGSHARE_API.format(id=article_id), timeout=60) as r:
        return json.load(r)


def _fetch(url: str, dst: Path, md5: str) -> None:
    if dst.exists() and _md5(dst) == md5:
        print(f"[skip] {dst}")
        return
    print(f"[get ] {dst}")
    tmp = dst.with_suffix(dst.suffix + ".part")
    urllib.request.urlretrieve(url, tmp)
    if _md5(tmp) != md5:
        tmp.unlink()
        raise RuntimeError(f"md5 mismatch: {dst}")
    tmp.rename(dst)


# CosMx NSCLC(He et al., Nat Biotechnol 2022),NanoString 公开 S3
COSMX_BASE = "https://nanostring-public-share.s3.us-west-2.amazonaws.com/SMI-Compressed"
COSMX_SAMPLES = ["Lung5_Rep1", "Lung5_Rep2", "Lung5_Rep3", "Lung6",
                 "Lung9_Rep1", "Lung9_Rep2", "Lung12", "Lung13"]


def download_cosmx(raw_root: Path | str = "data/raw/cosmx_nsclc") -> list[Path]:
    """下载 8 张切片的 Flat data(表达、元数据、坐标)和带 cell_type / niche 注释的 Giotto 对象。

    不下载原始形态学图像(约 360 GB)。
    """
    out_dir = Path(raw_root)
    out_dir.mkdir(parents=True, exist_ok=True)
    md5_path = out_dir / "md5sums.txt"
    urllib.request.urlretrieve(f"{COSMX_BASE}/md5sums.txt", md5_path)
    md5s = {line.split("  ", 1)[1].strip().removeprefix("./"): line.split()[0]
            for line in md5_path.read_text().splitlines() if line.strip()}

    rels = ["All SMI Giotto object.tar.gz"] + [f"{s}/{s} SMI Flat data.tar.gz" for s in COSMX_SAMPLES]
    paths = []
    for rel in rels:
        dst = out_dir / Path(rel).name
        _fetch(f"{COSMX_BASE}/{urllib.parse.quote(rel)}", dst, md5s[rel])
        paths.append(dst)
    return paths


def download_dataset(name: str, raw_root: Path | str = "data/raw/sdmbench") -> list[Path]:
    """下载一个数据集的全部 h5ad,按 md5 校验,已存在且校验通过的跳过。"""
    out_dir = Path(raw_root) / name
    out_dir.mkdir(parents=True, exist_ok=True)
    files = list_files(SDMBENCH_ARTICLES[name])
    (out_dir / "manifest.json").write_text(json.dumps(files, indent=2))

    paths = []
    for f in files:
        dst = out_dir / f["name"]
        _fetch(f["download_url"], dst, f["computed_md5"])
        paths.append(dst)
    return paths
