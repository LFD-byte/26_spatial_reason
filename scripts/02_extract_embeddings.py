"""冻结编码器提取每个细胞的嵌入,缓存到 data/embeddings/{encoder}/{dataset}/{sample}.npy。

    python scripts/02_extract_embeddings.py --encoder novae --datasets cosmx_nsclc merfish_hypothalamus starmap
"""

import argparse

from spatial_reason.data.datasets import list_samples, load_sample
from spatial_reason.encoders import get_encoder


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--encoder", default="novae")
    p.add_argument("--datasets", nargs="+", required=True)
    p.add_argument("--samples", nargs="*", default=None, help="默认处理该数据集的全部切片")
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args()

    enc = get_encoder(args.encoder)
    for ds in args.datasets:
        for s in args.samples or list_samples(ds):
            enc.encode_and_cache(load_sample(ds, s), ds, s, overwrite=args.overwrite)


if __name__ == "__main__":
    main()
