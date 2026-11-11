#!/usr/bin/env bash
# 在新服务器上重建 data/models 和 third_party(约 70 GB,主要是 Qwen3-32B)。可重复运行:hf download 会跳过已有文件。
#
#   bash scripts/download_models.sh            # 全部
#   bash scripts/download_models.sh --no-teacher   # 跳过 Qwen3-32B(只训练、不生成教师轨迹时)
#
# 国内网络可先 export HF_ENDPOINT=https://hf-mirror.com
set -euo pipefail
cd "$(dirname "$0")/.."
HF=${HF:-$HOME/software/anaconda3/envs/spatial_reason/bin/hf}
TEACHER=1
[ "${1:-}" = "--no-teacher" ] && TEACHER=0

M=data/models
mkdir -p "$M" third_party
get() { echo "== $1 -> $M/$2"; "$HF" download "$1" --local-dir "$M/$2"; }

get Qwen/Qwen3-4B Qwen3-4B
[ "$TEACHER" = 1 ] && get Qwen/Qwen3-32B Qwen3-32B
for v in human mouse brain; do get "prism-oncology/novae-$v-0" "novae-$v-0"; done
get theislab/Nicheformer Nicheformer

# 外部仓库(只读引用),commit 与当前实验一致
clone() {  # url dir commit
    [ -d "third_party/$2/.git" ] || git clone "$1" "third_party/$2"
    git -C "third_party/$2" checkout -q "$3"
}
clone https://github.com/bowang-lab/BioReason.git BioReason 684ff7c
clone https://github.com/wJDKnight/LLMiniST.git LLMiniST f5d4695
clone https://github.com/theislab/nicheformer.git nicheformer 485cadb

# Nicheformer 平台均值:HF 仓库里没有,取自官方 GitHub 的 data/model_means/
mkdir -p "$M/Nicheformer/technology_means"
cp third_party/nicheformer/data/model_means/*_mean_script.npy "$M/Nicheformer/technology_means/"

echo "[ok] 模型和外部代码就绪"
du -sh "$M"/*
