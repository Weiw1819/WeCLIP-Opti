#!/usr/bin/env bash

# 依次训练并测试 VOC 模型。
# 每次执行都会自动创建 output/exp1、output/exp2……，避免覆盖历史实验。

set -Eeuo pipefail

# 无论从哪个目录启动脚本，都切换到 WeCLIP 项目根目录运行。
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

CONDA_ENV="${CONDA_ENV:-cw}"
CONFIG_PATH="${1:-${SCRIPT_DIR}/configs/voc_attn_reg.yaml}"
OUTPUT_ROOT="${SCRIPT_DIR}/output"
EVAL_SET="${EVAL_SET:-val}"
BATCH_SIZE="${BATCH_SIZE:-}"

# 初始化 Conda，并激活用于运行本项目的 cw 虚拟环境。
CONDA_BIN="$(command -v conda || true)"
if [[ -z "${CONDA_BIN}" ]]; then
    echo "错误：未找到 conda，无法激活 ${CONDA_ENV} 虚拟环境。" >&2
    exit 1
fi

eval "$("${CONDA_BIN}" shell.bash hook)"
if ! conda activate "${CONDA_ENV}"; then
    echo "错误：无法激活 Conda 虚拟环境：${CONDA_ENV}" >&2
    exit 1
fi

PYTHON_BIN="${PYTHON_BIN:-python}"

if [[ ! -f "${CONFIG_PATH}" ]]; then
    echo "错误：配置文件不存在：${CONFIG_PATH}" >&2
    exit 1
fi

mkdir -p "${OUTPUT_ROOT}"

# 原子地创建下一个可用的实验目录，方便同时启动多个实验。
EXP_INDEX=1
while ! mkdir "${OUTPUT_ROOT}/exp${EXP_INDEX}" 2>/dev/null; do
    EXP_INDEX=$((EXP_INDEX + 1))
done
EXP_DIR="${OUTPUT_ROOT}/exp${EXP_INDEX}"

# 后续所有标准输出和错误输出既显示在终端，也保存到本次实验日志。
exec > >(tee -a "${EXP_DIR}/run.log") 2>&1

echo "实验目录：${EXP_DIR}"
echo "配置文件：${CONFIG_PATH}"
echo "Conda 环境：${CONDA_DEFAULT_ENV}"
echo "Python 路径：$(command -v "${PYTHON_BIN}")"
echo "开始训练：$(date '+%F %T')"

TRAIN_ARGS=(
    scripts/dist_clip_voc.py
    --config "${CONFIG_PATH}"
    --work_dir "${EXP_DIR}"
)
if [[ -n "${BATCH_SIZE}" ]]; then
    TRAIN_ARGS+=(--batch_size "${BATCH_SIZE}")
    echo "训练 batch size：${BATCH_SIZE}"
fi

"${PYTHON_BIN}" "${TRAIN_ARGS[@]}"

echo "训练结束：$(date '+%F %T')"

# 训练脚本会把不同时间启动的 checkpoint 放在 checkpoints/<时间戳>/ 下。
# 按文件名中的迭代次数排序，自动选择迭代次数最大的模型用于测试。
shopt -s nullglob
CHECKPOINTS=("${EXP_DIR}"/checkpoints/*/WeCLIP_model_iter_*.pth)
shopt -u nullglob

if (( ${#CHECKPOINTS[@]} == 0 )); then
    echo "错误：训练结束后未在 ${EXP_DIR}/checkpoints 中找到模型文件。" >&2
    exit 1
fi

MODEL_PATH="$(printf '%s\n' "${CHECKPOINTS[@]}" | sort -V | tail -n 1)"
TEST_OUTPUT_DIR="${EXP_DIR}/inference"

echo "测试模型：${MODEL_PATH}"
echo "开始测试：$(date '+%F %T')"

"${PYTHON_BIN}" test_msc_flip_voc.py \
    --config "${CONFIG_PATH}" \
    --model_path "${MODEL_PATH}" \
    --work_dir "${TEST_OUTPUT_DIR}" \
    --eval_set "${EVAL_SET}"

echo "测试结束：$(date '+%F %T')"
echo "本次实验全部结果已保存到：${EXP_DIR}"
