#!/usr/bin/env bash
# 把实验数据和结果同步到另一台服务器(代码走 git,不在这里同步)。
#
#   bash scripts/sync_project.sh user@host:/path/to/26_spatial_reason            # 默认:约 1 GB
#   bash scripts/sync_project.sh user@host:/path/... --with-models               # 再加模型权重 70 GB
#   bash scripts/sync_project.sh user@host:/path/... --with-raw                  # 再加原始数据 18 GB
#   bash scripts/sync_project.sh user@host:/path/... --dry-run                   # 只列出要传的文件
#   bash scripts/sync_project.sh myremote:bucket/26_spatial_reason --rclone      # 经云存储中转(两台机器不能直连时)
#
# 默认同步的是重新生成代价高或决定实验结果的部分:
#   data/processed  data/embeddings  data/traces  data/raw/gene_maps  outputs  third_party
# 模型可在新服务器上用 scripts/download_models.sh 从 HuggingFace 重新下载,通常比服务器间互传快;
# 原始数据只在需要重新预处理时才用得到(scripts/01_prepare_data.py 可重新下载)。
#
# rsync 增量 + 断点续传:可以先同步一次,训练结束后再跑一次,只传有变化的文件。
set -euo pipefail
cd "$(dirname "$0")/.."

[ $# -ge 1 ] || { sed -n 2,16p "$0"; exit 1; }
DEST=${1%/}; shift
WITH_MODELS=0; WITH_RAW=0; DRY=0; RCLONE=0
for a in "$@"; do
    case $a in
        --with-models) WITH_MODELS=1 ;;
        --with-raw) WITH_RAW=1 ;;
        --dry-run) DRY=1 ;;
        --rclone) RCLONE=1 ;;
        *) echo "未知参数 $a"; exit 1 ;;
    esac
done

PATHS=(data/processed data/embeddings data/traces data/raw/gene_maps outputs third_party
       data/README.md outputs/README.md)
[ "$WITH_MODELS" = 1 ] && PATHS+=(data/models)
[ "$WITH_RAW" = 1 ] && PATHS+=(data/raw)
EXCLUDES=(--exclude '*.tmp' --exclude '.pipeline.lock' --exclude '__pycache__' --exclude 'wandb/')

if pgrep -f "scripts/05_run_teacher_vllm.py|scripts/06_train.py" >/dev/null; then
    echo "[warn] 教师生成或训练正在运行:data/traces / outputs 还在变化,结束后需要再同步一次"
fi
if ! git diff --quiet || [ -n "$(git ls-files --others --exclude-standard)" ]; then
    echo "[warn] 有未提交的代码改动,不会被同步(代码请走 git)"
fi

echo "[sync] -> $DEST"
du -shc "${PATHS[@]}" 2>/dev/null | tail -1 | sed 's/total/ 合计(本地)/'

if [ "$RCLONE" = 1 ]; then
    RC=${RCLONE_BIN:-rclone}
    for p in "${PATHS[@]}"; do
        [ -e "$p" ] || continue
        if [ -d "$p" ]; then
            "$RC" copy "$p" "$DEST/$p" --progress --transfers 8 "${EXCLUDES[@]}" $([ "$DRY" = 1 ] && echo --dry-run)
        else
            "$RC" copyto "$p" "$DEST/$p" $([ "$DRY" = 1 ] && echo --dry-run)
        fi
    done
else
    EXIST=()
    for p in "${PATHS[@]}"; do [ -e "$p" ] && EXIST+=("./$p"); done
    # -R 保留相对路径结构;--partial 断网后续传;不加 --delete,目标端多出的文件不会被删
    rsync -aR --partial --info=progress2,stats1 "${EXCLUDES[@]}" $([ "$DRY" = 1 ] && echo --dry-run -v) \
        "${EXIST[@]}" "$DEST/"
fi
echo "[sync] 完成"
