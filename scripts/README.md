# scripts

命令行入口,只做参数解析和调用 `spatial_reason`,不放逻辑。按流程编号,计划如下:

| 脚本 | 作用 | 对应阶段(方案文档第 9 节) |
|---|---|---|
| `01_prepare_data.py` | 下载 / 预处理 / 划分 | 1 |
| `02_extract_embeddings.py` | 冻结编码器提取并缓存嵌入 | 1 |
| `03_run_baselines.py` | 线性头 / kNN、聚类、LLMiniST | 1 |
| `04_build_descriptions.py` | 自动生成邻域描述 | 2 |
| `05_generate_traces.py` | 教师轨迹生成与过滤 | 2 |
| `06_train.py` | `--config configs/train/stage{0,1,2}_*.yaml` | 2–3 |
| `07_evaluate.py` | 标签指标、开放词表、忠实性 | 4 |

集群提交脚本放 `scripts/slurm/`。
