#!/bin/bash
#SBATCH -p rnd,mts_ai,1gpu
#SBATCH --mem=80gb
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=32
#SBATCH --ntasks-per-node=1
#SBATCH --job-name serve_q3vl8b_think
#SBATCH --output=slurm/logs/%j.out

# Serves Qwen3-VL-8B-Thinking via vLLM with reasoning parser and thinking budget.
#
# Usage:
#   sbatch slurm/serve_q3vl8b_thinking.sbatch                      # defaults
#   PORT=8200 sbatch slurm/serve_q3vl8b_thinking.sbatch             # fixed port
#   THINK_BUDGET=512 sbatch slurm/serve_q3vl8b_thinking.sbatch      # custom budget
#   THINK_BUDGET=0 sbatch slurm/serve_q3vl8b_thinking.sbatch        # no budget
#   THINK_TRANSITION="Wrap up now." sbatch slurm/serve_q3vl8b_thinking.sbatch
#
# Per-request budget override (client-side):
#   extra_body={"vllm_xargs": {"think_budget": 256}}   # override for one request
#   extra_body={"vllm_xargs": {"think_budget": 0}}     # disable for one request

set -euo pipefail

MODEL=${MODEL:-'Qwen/Qwen3-VL-8B-Thinking'}
SERVED_NAME=${SERVED_NAME:-"${MODEL}"}
PORT=${PORT:-$(( RANDOM % 1000 + 8000 ))}
NODE=$(hostname)

# ── Think budget env vars (read by ThinkBudgetProcessor at startup) ──
export THINK_BUDGET=${THINK_BUDGET:-1024}
export THINK_TRANSITION="${THINK_TRANSITION:-}"          # empty = use Python default (includes \n</think>\n\n)
export THINK_END_TOKEN_ID=${THINK_END_TOKEN_ID:-}        # empty = auto-detect
export THINK_START_TOKEN_ID=${THINK_START_TOKEN_ID:-}    # empty = auto-detect

echo ">>> ${MODEL} endpoint: http://${NODE}:${PORT}/v1"
echo ">>> Served model name: ${SERVED_NAME}"
echo ">>> THINK_BUDGET=${THINK_BUDGET} (0=unlimited)"
[[ -n "${THINK_TRANSITION}" ]] && echo ">>> THINK_TRANSITION='${THINK_TRANSITION}'" || echo ">>> THINK_TRANSITION=(default from Qwen3 paper)"
echo ">>>"
echo ">>> Per-request override:"
echo ">>>   extra_body={\"vllm_xargs\": {\"think_budget\": 256}}"

LOGITS_ARG=""
if [[ "${THINK_BUDGET}" -gt 0 ]]; then
    LOGITS_ARG="--logits-processors think_budget:ThinkBudgetProcessor"
fi

# Build container env list (pyxis imports these by name from host env)
CONTAINER_ENV="THINK_BUDGET"
[[ -n "${THINK_TRANSITION}" ]] && CONTAINER_ENV="${CONTAINER_ENV},THINK_TRANSITION"
[[ -n "${THINK_END_TOKEN_ID}" ]] && CONTAINER_ENV="${CONTAINER_ENV},THINK_END_TOKEN_ID"
[[ -n "${THINK_START_TOKEN_ID}" ]] && CONTAINER_ENV="${CONTAINER_ENV},THINK_START_TOKEN_ID"

mkdir -p slurm/logs

REPO_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"

srun \
--container-image=vllm/vllm-openai:v0.11.0 \
--container-mounts=${HOME}:${HOME},/share:/share \
--container-name="serve_q3vl8b_think_${PORT}_${SLURM_JOB_ID}" \
--container-env="${CONTAINER_ENV}" \
bash -c "
  pip install --break-system-packages ${REPO_DIR} && \
  python3 -c 'import think_budget; print(f\"think_budget {think_budget.__version__} loaded from {think_budget.__file__}\")' && \
  python3 -m vllm.entrypoints.openai.api_server \
      --host 0.0.0.0 \
      --port ${PORT} \
      --model ${MODEL} \
      --served-model-name '${SERVED_NAME}' \
      --max-model-len 32000 \
      --tensor-parallel-size 1 \
      --gpu-memory-utilization 0.90 \
      --enable-auto-tool-choice \
      --tool-call-parser hermes \
      --reasoning-parser qwen3 \
      ${LOGITS_ARG}
"
