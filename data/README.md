# data(不进 git)

- `raw/`:原始下载,只读
- `processed/`:统一格式的 AnnData(.h5ad)与划分文件
- `embeddings/`:各编码器的缓存嵌入,`{encoder}/{dataset}/`
- `traces/`:教师推理轨迹(原始与过滤后)

数据量大时可把本目录做成指向共享存储的软链接。
