# project_spatial_reason

空间嵌入 + LLM 推理(目标:ACL 2027)。方案见 [docs/方案定位与实验路线.md](docs/方案定位与实验路线.md)。

## 目录

```
├── docs/            方案文档、会议记录
├── configs/         YAML 配置:data / encoder / model / train / eval / ablation
├── src/spatial_reason/
│   ├── data/        加载、划分、邻域构造、自动描述、抽象标签
│   ├── encoders/    冻结编码器:Novae / Nicheformer / PCA
│   ├── models/      投影层、坐标编码、SpatialLLM
│   ├── traces/      教师推理轨迹生成与过滤
│   ├── training/    阶段 0 对齐 / 阶段 1 SFT / 阶段 2 GRPO
│   ├── rewards/     标签、格式、证据、空间一致性
│   ├── inference/   生成、解析、原型修正
│   ├── baselines/   线性头、纯文本 Qwen、空间聚类、LLMiniST
│   ├── evaluation/  指标、开放词表、忠实性、富集判定
│   └── utils/
├── scripts/         编号的命令行入口
├── tests/
├── notebooks/
├── paper/           论文源文件
├── data/            数据与缓存(不进 git)
├── outputs/         检查点、日志、结果(不进 git)
└── third_party/     BioReason、LLMiniST 等外部代码(不进 git)
```

## 安装

换服务器见 [docs/迁移服务器.md](docs/迁移服务器.md):代码走 git,数据用 `scripts/sync_project.sh`,模型用 `scripts/download_models.sh`,环境按 `envs/*.lock.txt` 重建。

依赖声明在 `pyproject.toml`,不绑定 pixi:

```bash
pixi install                      # 或
pip install -e ".[llm,encoders,dev]"
```
