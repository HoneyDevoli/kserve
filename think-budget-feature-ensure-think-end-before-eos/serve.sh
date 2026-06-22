#!/bin/bash
# Local vLLM server with thinking budget.
#
# Usage:
#   ./serve.sh                                          # defaults (Qwen3.5-9B, budget=1024)
#   MODEL=Qwen/Qwen3-VL-8B-Thinking ./serve.sh         # different model
#   THINK_BUDGET=512 ./serve.sh                         # custom budget
#   THINK_BUDGET=0 ./serve.sh                           # no budget
#   PORT=8200 ./serve.sh                                # custom port
#   PARSER=hermes ./serve.sh                            # hermes parser (Qwen3-VL)
#   THINK_TRANSITION="Wrap up." ./serve.sh              # custom transition
#
# Per-request budget override (client-side):
#   extra_body={"vllm_xargs": {"think_budget": 256}}
#
# Thinking control (Qwen3.5 hybrid):
#   ON:  extra_body={"chat_template_kwargs": {"enable_thinking": true}}
#   OFF: extra_body={"chat_template_kwargs": {"enable_thinking": false}}

set -euo pipefail

MODEL=${MODEL:-'Qwen/Qwen3.5-9B'}
PORT=${PORT:-8150}
TP=${TP:-1}
MAX_LEN=${MAX_LEN:-32000}
GPU_UTIL=${GPU_UTIL:-0.90}
PARSER=${PARSER:-qwen3_coder}     # qwen3_coder (Qwen3.5) or hermes (Qwen3-VL)

# ── Think budget ──
export THINK_BUDGET=${THINK_BUDGET:-1024}
export THINK_TRANSITION="${THINK_TRANSITION:-}"   # empty = use Python default (includes \n</think>\n\n)
export THINK_END_TOKEN_ID=${THINK_END_TOKEN_ID:-}
export THINK_START_TOKEN_ID=${THINK_START_TOKEN_ID:-}

# ── Install think-budget if needed ──
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
python3 -c "import think_budget" 2>/dev/null || pip install -q -e "${SCRIPT_DIR}"

echo ">>> Model: ${MODEL}"
echo ">>> Endpoint: http://localhost:${PORT}/v1"
echo ">>> THINK_BUDGET=${THINK_BUDGET} (0=disabled)"
[[ -n "${THINK_TRANSITION}" ]] && echo ">>> THINK_TRANSITION='${THINK_TRANSITION}'" || echo ">>> THINK_TRANSITION=(default from Qwen3 paper)"
echo ">>> Parser: ${PARSER}, TP=${TP}, max_len=${MAX_LEN}"

LOGITS_ARG=""
if [[ "${THINK_BUDGET}" -gt 0 ]]; then
    LOGITS_ARG="--logits-processors think_budget:ThinkBudgetProcessor"
fi

python3 -m vllm.entrypoints.openai.api_server \
    --host 0.0.0.0 \
    --port ${PORT} \
    --model ${MODEL} \
    --served-model-name ${MODEL} \
    --max-model-len ${MAX_LEN} \
    --tensor-parallel-size ${TP} \
    --gpu-memory-utilization ${GPU_UTIL} \
    --reasoning-parser qwen3 \
    --enable-auto-tool-choice \
    --tool-call-parser ${PARSER} \
    ${LOGITS_ARG}
