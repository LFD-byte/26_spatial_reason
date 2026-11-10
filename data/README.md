# data(不进 git)

- `raw/`:原始下载,只读
- `processed/`:统一格式的 AnnData(.h5ad)与划分文件
- `embeddings/`:各编码器的缓存嵌入,`{encoder}/{dataset}/`
- `traces/`:教师推理轨迹(原始与过滤后)

数据量大时可把本目录做成指向共享存储的软链接。

## 来源与生成

`python scripts/01_prepare_data.py` 一步完成下载(md5 校验)和标准化。

| 数据集 | 来源 | 切片 | 备注 |
|---|---|---|---|
| dlpfc | SDMBench Figshare 22548901 | 12(3 个供体) | 全转录组原始计数;`cell_type` 为空,需反卷积;Br5595 只有 Layer3–6 + WM |
| merfish_hypothalamus | SDMBench Figshare 22565170 | 5(Animal 1,Bregma -0.04…-0.24) | X 为非整数(原文按细胞体积归一化) |
| starmap | SDMBench Figshare 22565200 | 3(BZ5/9/14) | 域标签为数字编码 1–4,与 LLMiniST 自带示例一致 |
| cosmx_nsclc | NanoString 公开 S3 `SMI-Compressed/`(He et al. 2022) | 8(5 个病人) | 960 基因原始计数;`cell_type` 22 类、`domain` = niche 9 类,来自 Giotto 对象;坐标单位 µm |

`processed/{dataset}/{sample}.h5ad` 统一字段:`obs[cell_type, domain, sample, donor, platform]`、`obsm['spatial']`,格式说明见 `src/spatial_reason/data/prepare.py`。

CosMx 额外步骤:Giotto 对象(.RData)用 `scripts/r/export_cosmx_giotto.R` 导出为 `raw/cosmx_nsclc/cell_annotations.csv`
(无需安装 Giotto)。Flat data 只解出 exprMat / metadata / fov_positions,跳过转录本文件和图像。

## 模型权重 `models/`

| 目录 | 来源 | 用途 |
|---|---|---|
| `novae-human-0` | HF `prism-oncology/novae-human-0` | CosMx、DLPFC |
| `novae-mouse-0` | HF `prism-oncology/novae-mouse-0` | MERFISH、STARmap |
| `novae-brain-0` | HF `prism-oncology/novae-brain-0` | 脑组织对比 |
| `Qwen3-4B` | HF `Qwen/Qwen3-4B` | LLM(hidden 2560) |
| `Nicheformer` | HF `theislab/Nicheformer` + GitHub `data/model_means/` → `technology_means/` | 对比编码器(512 维) |

基因映射表 `raw/gene_maps/`:HGNC complete set(人类 symbol → Ensembl)、MGI `HOM_MouseHumanSequence.rpt`
(小鼠 → 人类 1:1 直系同源)。Ensembl BioMart 当时不可用,故用这两份。

注意:
- Novae 把非整数矩阵当作已 log 化。MERFISH 的 X 不是整数,需先 `normalize_total + log1p`,否则 HVG 步骤溢出为 inf。
- Novae 的 `radius` 与坐标单位一致:STARmap 是像素(中位最近邻距离约 200),其余为 µm。
- 本机驱动 535(CUDA 12.2):torch 需用 cu126 版本(`--index-url https://download.pytorch.org/whl/cu126`),
  默认 PyPI 的 cu13x 版本无法使用 GPU。

Nicheformer(`src/spatial_reason/encoders/nicheformer.py`):
- 基因需对齐到参考词表(20310 个人类 Ensembl);HF 自带 tokenizer 在本地路径下会静默跳过对齐,不要用。
- 平台均值文件对未测基因为 NaN。官方实现中这些基因会作为固定尾巴接在每个细胞的 token 序列后(`nan_mode="official"`,默认);
  `nan_mode="drop"` 丢弃它们。kNN 细胞类型准确率:CosMx 0.71 / 0.71,MERFISH 0.66 / 0.77。
- STARmap 与 Visium 不在其预训练平台内(用近似 token 和均值);STARmap 只有约 9 个基因有 iss 均值,嵌入几乎无信息。
- **数据重叠**:Nicheformer 官方 notebook 含 `cosmx_human_lung`(同样的 9 个 niche),CosMx NSCLC 很可能在其预训练语料
  SpatialCorpus-110M 中;论文需说明。
- 速度约 40 细胞/秒/GPU(batch 32,序列 1500),CosMx 全量 77 万细胞需要提速(半精度、双卡)。
