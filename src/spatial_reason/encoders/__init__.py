"""冻结的空间编码器,统一接口:AnnData -> 每细胞嵌入。"""

from spatial_reason.encoders import novae, pca  # noqa: F401  注册 "novae" / "novae_ft" / "pca"
from spatial_reason.encoders.base import get_encoder, load_embeddings

__all__ = ["get_encoder", "load_embeddings"]
