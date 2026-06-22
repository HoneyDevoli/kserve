#!/bin/bash
#SBATCH -p rnd,mts_ai,1gpu
#SBATCH --mem=60gb
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --ntasks-per-node=1
#SBATCH --job-name serve_q35_think_[COT-1567]
#SBATCH --output=slurm/logs/%j.out

# Serves Qwen3.5 (hybrid reasoner) via vLLM with server-side thinking budget.
#
# Qwen3.5 can think or not think — controlled per-request via enable_thinking.
# The ThinkBudgetProcessor only activates when the model generates <think>.
#
# Usage:
#   sbatch slurm/serve_q35_thinking.sbatch                          # defaults
#   PORT=8200 sbatch slurm/serve_q35_thinking.sbatch                 # fixed port
#   THINK_BUDGET=512 sbatch slurm/serve_q35_thinking.sbatch          # custom budget
#   THINK_BUDGET=0 sbatch slurm/serve_q35_thinking.sbatch            # no budget
#   MODEL=Qwen/Qwen3.5-27B sbatch slurm/serve_q35_thinking.sbatch   # larger model
#   THINK_TRANSITION="Wrap up now." sbatch slurm/serve_q35_thinking.sbatch
#
# Client-side thinking control:
#   Thinking ON:  extra_body={"chat_template_kwargs": {"enable_thinking": true}}
#   Thinking OFF: extra_body={"chat_template_kwargs": {"enable_thinking": false}}
#
# Per-request budget override (client-side):
#   extra_body={"chat_template_kwargs": {"enable_thinking": true}, "vllm_xargs": {"think_budget": 256}}
#   extra_body={"chat_template_kwargs": {"enable_thinking": true}, "vllm_xargs": {"think_budget": 0}}

set -euo pipefail

MODEL=${MODEL:-'Qwen/Qwen3.5-9B'}
PORT=${PORT:-$(( RANDOM % 1000 + 8000 ))}
NODE=$(hostname)

# ── Think budget env vars (read by ThinkBudgetProcessor at startup) ──
export THINK_BUDGET=${THINK_BUDGET:-1024}
export THINK_TRANSITION="${THINK_TRANSITION:-}"          # empty = use Python default (includes \n</think>\n\n)
export THINK_END_TOKEN_ID=${THINK_END_TOKEN_ID:-}        # empty = auto-detect
export THINK_START_TOKEN_ID=${THINK_START_TOKEN_ID:-}    # empty = auto-detect

echo ">>> ${MODEL} + ThinkBudget endpoint: http://${NODE}:${PORT}/v1"
echo ">>> THINK_BUDGET=${THINK_BUDGET} (0=disabled)"
[[ -n "${THINK_TRANSITION}" ]] && echo ">>> THINK_TRANSITION='${THINK_TRANSITION}'" || echo ">>> THINK_TRANSITION=(default from Qwen3 paper)"
echo ">>>"
echo ">>> Reasoning ON (pass enable_thinking=true via extra_body):"
echo ">>>   extra_body={\"chat_template_kwargs\": {\"enable_thinking\": true}}"
echo ">>>"
echo ">>> Reasoning OFF (pass enable_thinking=false, or omit — model default):"
echo ">>>   extra_body={\"chat_template_kwargs\": {\"enable_thinking\": false}}"
echo ">>>"
echo ">>> Per-request budget override:"
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
--container-image=vllm/vllm-openai:nightly \
--container-mounts=${HOME}:${HOME},/share:/share \
--container-name="serve_q35_think_${PORT}_${SLURM_JOB_ID}" \
--container-env="${CONTAINER_ENV}" \
bash -c "
  pip install --break-system-packages ${REPO_DIR} && \
  python3 -c 'import think_budget; print(f\"think_budget {think_budget.__version__} loaded from {think_budget.__file__}\")' && \
  python3 -m vllm.entrypoints.openai.api_server \
      --host 0.0.0.0 \
      --port ${PORT} \
      --model ${MODEL} \
      --served-model-name ${MODEL} \
      --max-model-len 32000 \
      --tensor-parallel-size 1 \
      --gpu-memory-utilization 0.90 \
      --reasoning-parser qwen3 \
      --enable-auto-tool-choice \
      --tool-call-parser qwen3_coder \
      ${LOGITS_ARG}
"
