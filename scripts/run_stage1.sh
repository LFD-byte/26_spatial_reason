#!/usr/bin/env bash
# 阶段 1 全流程,可重入:停机 / 中断后重跑同一条命令即可,已完成的步骤自动跳过,未完成的从断点继续。
#
#   nohup bash scripts/run_stage1.sh > logs/stage1_pipeline.log 2>&1 &
#
#   1. 构造教师请求        requests_{train,val}.jsonl 已存在则跳过
#   2. 教师生成(vLLM)     每 chunk 落盘,按 id 跳过已完成的请求
#   3. 过滤轨迹            sft_*.jsonl 比 teacher_*.jsonl 新则跳过
#   4. SFT(双卡 DDP)      output_dir 有 DONE 则跳过;有 checkpoint.pt 则续训
#
# 可用环境变量覆盖:CONFIG, TRACES_DIR, TEACHER_MODEL, VLLM_PY, TRAIN_ENV
set -euo pipefail
cd "$(dirname "$0")/.."

CONFIG=${CONFIG:-configs/train/stage1_sft.yaml}
TRACES_DIR=${TRACES_DIR:-data/traces/qwen3_32b}
TEACHER_MODEL=${TEACHER_MODEL:-data/models/Qwen3-32B}
VLLM_PY=${VLLM_PY:-$HOME/software/anaconda3/envs/vllm/bin/python}
TRAIN_ENV=${TRAIN_ENV:-$HOME/software/anaconda3/envs/spatial_reason}
PY="$TRAIN_ENV/bin/python"
export PYTHONPATH=src${PYTHONPATH:+:$PYTHONPATH}

mkdir -p "$TRACES_DIR"
# 同一时间只允许一个流水线(防止重复启动抢 GPU)
exec 9>"$TRACES_DIR/.pipeline.lock"
flock -n 9 || { echo "[pipeline] 已有一个流水线在运行,退出"; exit 1; }
if pgrep -f "scripts/05_run_teacher_vllm.py" >/dev/null || pgrep -f "scripts/06_train.py" >/dev/null; then
    echo "[pipeline] 检测到流水线外的教师生成或训练进程在运行,先等它结束或手动停止"; exit 1
fi

ts() { date "+%F %T"; }
lines() { [ -f "$1" ] && wc -l < "$1" || echo 0; }

echo "[$(ts)] 1/4 教师请求"
if [ -f "$TRACES_DIR/requests_train.jsonl" ] && [ -f "$TRACES_DIR/requests_val.jsonl" ]; then
    echo "  已存在,跳过"
else
    "$PY" scripts/05_build_teacher_prompts.py --config "$CONFIG"
fi

echo "[$(ts)] 2/4 教师生成"
for split in val train; do
    req="$TRACES_DIR/requests_$split.jsonl"; out="$TRACES_DIR/teacher_$split.jsonl"
    if [ "$(lines "$out")" -ge "$(lines "$req")" ]; then
        echo "  $split 已完成 ($(lines "$out")/$(lines "$req")),跳过"; continue
    fi
    # 两卡之间无 P2P(nvidia-smi topo 为 SYS):不加这几个选项会卡死或非法访存
    NCCL_P2P_DISABLE=1 "$VLLM_PY" scripts/05_run_teacher_vllm.py --model "$TEACHER_MODEL" --tp 2 --n 2 \
        --temperature 0.7 --max-tokens 640 --requests "$req" --output "$out" \
        --enforce-eager --disable-custom-all-reduce --gpu-memory-utilization 0.85
    [ "$(lines "$out")" -ge "$(lines "$req")" ] || { echo "  $split 未完成,退出"; exit 1; }
done

echo "[$(ts)] 3/4 过滤轨迹"
if [ -f "$TRACES_DIR/sft_train.jsonl" ] && [ "$TRACES_DIR/sft_train.jsonl" -nt "$TRACES_DIR/teacher_train.jsonl" ] \
   && [ "$TRACES_DIR/sft_val.jsonl" -nt "$TRACES_DIR/teacher_val.jsonl" ]; then
    echo "  已是最新,跳过"
else
    "$PY" scripts/05_filter_traces.py --dir "$TRACES_DIR"
fi

echo "[$(ts)] 4/4 Stage 1 SFT(有 checkpoint 自动续训)"
sleep 10  # 等 vLLM 释放显存
"$TRAIN_ENV/bin/accelerate" launch --config_file configs/accelerate/ddp_2gpu.yaml \
    scripts/06_train.py --config "$CONFIG" --override "traces.dir=$TRACES_DIR"

echo "[$(ts)] 全部完成"
