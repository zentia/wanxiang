#!/usr/bin/env bash
# Rebuild all 6 Zhihu covers from ONE textless AI background.
# No image-generation credits are consumed here: this only re-composites
# the Chinese title layer with Pillow. Safe to re-run any time.
#
#   bash zhihu-series/covers/build_all.sh
#
# All 6 use the same palette/layout (make_cover.py); only the text and the
# issue number differ, which keeps the series visually unified.
set -euo pipefail

cd "$(dirname "$0")/../../"          # repo root
PY=".venv/bin/python"
BG="zhihu-series/covers/bg_01.png"
OUT="zhihu-series/covers"
MK="zhihu-series/covers/make_cover.py"
KICKER="从零训练一个大模型 · 硬核工程复盘"
FOOTER="WANXIANG · 自研大模型工程复盘连载"

cover() {
  local no="$1"; shift
  "$PY" "$MK" --bg "$BG" --out "$OUT/cover_${no}.png" \
    --no "$no" --total 06 --kicker "$KICKER" --footer "$FOOTER" "$@"
}

# 01 — 立项 / 定位漂移（样张，同一参数重出，保证 6 张完全一致）
cover 01 \
  --line1 "我想自己训一个大模型" \
  --line2-seg "前三周:accent" "都在吵它该干什么:white" \
  --sub "立项与定位的三次漂移 · 弯路复盘"

# 02 — 造尺子 / 训练基础设施（高亮词前移，标题收在左侧暗部）
cover 02 \
  --line1 "动手写模型之前" \
  --line2-seg "造尺子:accent" "比造模型更急:white" \
  --sub "配置哈希守卫 · 冻结评测 · 会拒绝打分"

# 03 — 测量污染 / 撤回结论
cover 03 \
  --line1 "我把模型训好了35%" \
  --line2-seg "亲手撤回:accent" "那个结论:white" \
  --sub "留出集污染 · 记忆与泛化的分离"

# 04 — 三路撞墙 / 任务性质错配
cover 04 \
  --line1 "加长·换词·扩容量" \
  --line2-seg "三条路:accent" "撞同一堵墙:white" \
  --sub "不是超参问题 · 是任务性质错配"

# 05 — 方向纠偏 / 通用 LM 主线
cover 05 \
  --line1 "推倒重来，收敛目标" \
  --line2-seg "只留:white" "单纯的大模型:accent" \
  --sub "通用语料 + 困惑度尺子 · 训练核心零改动"

# 06 — 主线严格 A/B / 硬核收尾
cover 06 \
  --line1 "用同一把尺子量到底" \
  --line2-seg "困惑度:white" "从304打到56:accent" \
  --sub "加数据/步数 · 扩容量 · 同 hash BPE 三路 A/B"

echo "[done] 6 covers in $OUT"
