# think-budget

vLLM logits processor for controlling the thinking token budget in reasoning models (Qwen3, Qwen3.5, DeepSeek-R1, etc.).

When the model exceeds the token budget inside a `<think>...</think>` block, the processor injects a transition instruction (from the [Qwen3 paper](https://arxiv.org/abs/2505.09388), Section 3.4) and forces `</think>`, guiding the model to produce a coherent answer based on its accumulated partial reasoning.

**Requires**: vLLM >= 0.11.0 (V1 engine)

## How it works

```
<think>
[model reasons freely until budget is reached]
Considering the limited time by the user, I have to give the   ← injected
solution based on the thinking directly now.                    ← injected
</think>                                                        ← forced

[model produces answer based on partial reasoning]
```

Without the transition instruction, abruptly forcing `</think>` leads to worse answers because the model has no signal to wrap up. The transition text (from the Qwen3 paper) naturally emerges in Thinking-mode-trained models and produces significantly better outputs from incomplete reasoning.

**Measured overhead**: the transition instruction adds exactly **22 tokens** to the thinking output beyond the budget. For example, with `THINK_BUDGET=64`, reasoning output is ~86 tokens.

| Budget | Reasoning tokens | Overshoot | Transition injected |
|--------|-----------------|-----------|---------------------|
| 64     | 86              | 22        | Yes                 |
| 128    | 150             | 22        | Yes                 |
| 256    | 278             | 22        | Yes                 |
| 512    | 534             | 22        | Yes                 |
| 1024   | 1046            | 22        | Yes                 |

## Install

```bash
pip install /path/to/think-budget

# Or editable (for development)
pip install -e /path/to/think-budget
```

## Usage

### Quick start (local)

```bash
# Default: Qwen3.5-9B, budget=1024, port=8150
./serve.sh

# Custom model and budget
MODEL=Qwen/Qwen3-VL-8B-Thinking THINK_BUDGET=512 PARSER=hermes ./serve.sh

# Qwen3.5-27B on 2 GPUs
MODEL=Qwen/Qwen3.5-27B TP=2 THINK_BUDGET=2048 ./serve.sh

# No budget (unlimited thinking)
THINK_BUDGET=0 ./serve.sh

# Custom transition phrase
THINK_TRANSITION="Time is up, answer now." ./serve.sh
```

### Manual vLLM CLI

```bash
# Qwen3-VL-8B-Thinking (always-thinking model)
THINK_BUDGET=512 python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen3-VL-8B-Thinking \
    --reasoning-parser qwen3 \
    --tool-call-parser hermes \
    --enable-auto-tool-choice \
    --logits-processors think_budget:ThinkBudgetProcessor \
    --host 0.0.0.0 --port 8143

# Qwen3.5-9B (hybrid reasoner — can think or not think per request)
THINK_BUDGET=1024 python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen3.5-9B \
    --reasoning-parser qwen3 \
    --tool-call-parser qwen3_coder \
    --enable-auto-tool-choice \
    --logits-processors think_budget:ThinkBudgetProcessor \
    --host 0.0.0.0 --port 8160
```

### SLURM (with pyxis/enroot containers)

Example SLURM scripts are in `slurm/`:

```bash
# Qwen3-VL-8B-Thinking
sbatch slurm/serve_q3vl8b_thinking.sbatch
THINK_BUDGET=512 sbatch slurm/serve_q3vl8b_thinking.sbatch

# Qwen3.5 (hybrid)
sbatch slurm/serve_q35_thinking.sbatch
THINK_BUDGET=512 MODEL=Qwen/Qwen3.5-27B sbatch slurm/serve_q35_thinking.sbatch
```

The SLURM scripts use `vllm/vllm-openai` Docker images via pyxis. The `think-budget` package is installed inside the container at startup with `pip install --break-system-packages`. Environment variables (`THINK_BUDGET`, etc.) are passed to the container via `--container-env`.

### Client (OpenAI SDK)

The processor is transparent to clients — just call the endpoint normally.

#### Qwen3-VL (always-thinking model)

```python
from openai import OpenAI

client = OpenAI(base_url="http://hostname:8143/v1", api_key="EMPTY")

resp = client.chat.completions.create(
    model="Qwen/Qwen3-VL-8B-Thinking",
    messages=[{"role": "user", "content": "What is 15 * 37?"}],
    extra_body={"chat_template_kwargs": {"enable_thinking": True}},
    max_tokens=4096,
    temperature=0.6,
)

# vLLM returns reasoning in a separate field
msg = resp.choices[0].message
reasoning = msg.model_dump().get("reasoning", "")
answer = msg.content
print(f"Reasoning: {reasoning}")
print(f"Answer: {answer}")
```

#### Qwen3.5 (hybrid reasoner — enable/disable thinking per request)

Qwen3.5 is a hybrid model: thinking is controlled via `enable_thinking` in `chat_template_kwargs`. The budget processor automatically detects whether thinking is enabled by inspecting the prompt tokens — safe for both modes without any extra configuration.

```python
client = OpenAI(base_url="http://hostname:8160/v1", api_key="EMPTY")

# Thinking ON — model reasons inside <think>...</think>, budget enforced
resp = client.chat.completions.create(
    model="Qwen/Qwen3.5-9B",
    messages=[{"role": "user", "content": "Solve step by step: ..."}],
    extra_body={"chat_template_kwargs": {"enable_thinking": True}},
    max_tokens=4096,
    temperature=0.7,
)

# Thinking OFF — model answers directly, processor auto-detects and skips
resp = client.chat.completions.create(
    model="Qwen/Qwen3.5-9B",
    messages=[{"role": "user", "content": "What is the capital of France?"}],
    extra_body={"chat_template_kwargs": {"enable_thinking": False}},
    max_tokens=4096,
    temperature=0.7,
)
```

#### With tool use

```python
tools = [
    {"type": "function", "function": {
        "name": "get_company_info",
        "description": "Get company info by INN",
        "parameters": {"type": "object", "properties": {"inn": {"type": "string"}}, "required": ["inn"]}
    }}
]

# Qwen3.5 with thinking + tools
resp = client.chat.completions.create(
    model="Qwen/Qwen3.5-9B",
    messages=[
        {"role": "system", "content": "You are a tax service operator."},
        {"role": "user", "content": "Check company INN 7701234567"},
    ],
    tools=tools,
    extra_body={"chat_template_kwargs": {"enable_thinking": True}},
    max_tokens=4096,
)
msg = resp.choices[0].message
if msg.tool_calls:
    for tc in msg.tool_calls:
        print(f"Tool: {tc.function.name}({tc.function.arguments})")
```

### Per-request budget override

Override the server-level budget for individual requests via `vllm_xargs`:

```python
# Use a smaller budget for this specific request
resp = client.chat.completions.create(
    model="Qwen/Qwen3.5-9B",
    messages=[{"role": "user", "content": "Quick answer: ..."}],
    extra_body={
        "chat_template_kwargs": {"enable_thinking": True},
        "vllm_xargs": {"think_budget": 256},
    },
)

# Disable budget for this request (unlimited thinking)
resp = client.chat.completions.create(
    model="Qwen/Qwen3.5-9B",
    messages=[{"role": "user", "content": "Think carefully: ..."}],
    extra_body={
        "chat_template_kwargs": {"enable_thinking": True},
        "vllm_xargs": {"think_budget": 0},
    },
)
```

### Ensure `</think>` before EOS

Some reasoning models try to stop while they are still inside the
`<think>...</think>` block and never emit `</think>`.  Enable the EOS guard to
prevent premature EOS before reasoning is closed:

```bash
THINK_ENSURE_END_BEFORE_EOS=1 THINK_EOS_PROB_THRESHOLD=0.2 ./serve.sh
```

Recommended starting point:

```bash
THINK_ENSURE_END_BEFORE_EOS=1 \
THINK_EOS_PROB_THRESHOLD=0.2 \
THINK_BAN_EOS_AFTER_THINK_END_TOKENS=1 \
./serve.sh
```

`0.2` is intentionally conservative: it still always triggers when EOS is
argmax, and additionally triggers when EOS already has a strong softmax
probability signal.  Use `0.1` for more aggressive early closure, or `0.3` for
a more cautious guard.

When `</think>` is not present in generated output, the processor:

1. Reads the original EOS probability from the input logits.
2. If EOS is argmax, or if `P(EOS) >= THINK_EOS_PROB_THRESHOLD`, forces the same
   transition instruction used by the budget limiter, including `</think>`.
3. Otherwise masks EOS by setting its logit to `-inf`, so sampled decoding cannot
   terminate before `</think>`.

Per-request override:

```python
resp = client.chat.completions.create(
    model="Qwen/Qwen3.5-9B",
    messages=[{"role": "user", "content": "Think and answer: ..."}],
    extra_body={
        "chat_template_kwargs": {"enable_thinking": True},
        "vllm_xargs": {
            "ensure_think_end_before_eos": True,
            "think_eos_prob_threshold": 0.2,
        },
    },
)
```

You can also require at least one answer token after a forced `</think>`:

```bash
THINK_ENSURE_END_BEFORE_EOS=1 THINK_BAN_EOS_AFTER_THINK_END_TOKENS=1 ./serve.sh
```

or per request:

```python
"vllm_xargs": {
    "ensure_think_end_before_eos": True,
    "ban_eos_after_think_end_tokens": 1,
}
```

`THINK_BUDGET=0` disables budget enforcement, but the EOS guard still works if
`THINK_ENSURE_END_BEFORE_EOS=1`.

### Offline inference (`vllm.LLM`)

For batch inference without a server:

```python
from vllm import LLM, SamplingParams
from think_budget import make_think_budget_processor

llm = LLM(model="Qwen/Qwen3.5-9B")
tokenizer = llm.get_tokenizer()

# Create processor (auto-detects token IDs from tokenizer)
processor = make_think_budget_processor(tokenizer, budget=512)

# Custom transition phrase
processor = make_think_budget_processor(
    tokenizer, budget=1024,
    transition="Time is up, give your answer now.",
)

params = SamplingParams(
    temperature=0.7,
    max_tokens=4096,
    logits_processors=[processor],
)

messages = [{"role": "user", "content": "Solve: 123 * 456"}]
prompt = tokenizer.apply_chat_template(
    messages, tokenize=False,
    add_generation_prompt=True,
    enable_thinking=True,  # Qwen3.5: enable thinking
)
outputs = llm.generate([prompt], sampling_params=params)
print(outputs[0].outputs[0].text)
```

With tools:

```python
tools = [{"type": "function", "function": {"name": "search", ...}}]

prompt = tokenizer.apply_chat_template(
    messages, tools=tools, tokenize=False,
    add_generation_prompt=True,
    enable_thinking=True,
)
outputs = llm.generate([prompt], sampling_params=params)
```

## Environment variables

| Variable | Default | Description |
|---|---|---|
| `THINK_BUDGET` | `1024` | Max thinking tokens per request. `0` = disabled (unlimited). |
| `THINK_END_TOKEN_ID` | auto-detected | Override `</think>` token ID. Auto-detected from model tokenizer at startup. Falls back to `151668` (Qwen3). |
| `THINK_START_TOKEN_ID` | auto-detected | Override `<think>` token ID. Auto-detected from model tokenizer at startup. |
| `THINK_EOS_TOKEN_ID` | auto-detected | Override EOS token ID. Auto-detected from tokenizer `eos_token_id`. |
| `THINK_TRANSITION` | *(Qwen3 paper text)* | Override transition instruction text injected when budget is reached. Default includes `\n</think>\n\n` suffix. |
| `THINK_ENSURE_END_BEFORE_EOS` | `0` | If enabled, prevent EOS before `</think>` and force transition when EOS is argmax or above threshold. |
| `THINK_EOS_PROB_THRESHOLD` | `0.0` | Softmax probability threshold for forcing `</think>` before EOS. `0.0` means argmax trigger only. |
| `THINK_BAN_EOS_AFTER_THINK_END_TOKENS` | `0` | Ban EOS for this many answer tokens after a forced `</think>`. `1` guarantees a non-empty answer token. |

## Architecture

The processor has two layers:

1. **`ThinkBudgetProcessor`** (server-level) — subclasses vLLM's `AdapterLogitsProcessor`. Loaded once at server startup via `--logits-processors` CLI arg. Reads config from env vars, auto-detects token IDs, pre-tokenizes the transition instruction.

2. **`_ThinkBudgetReqProcessor`** (per-request) — instantiated for each incoming request. Tracks output tokens and enforces the budget:
   - `len(output) < budget` → pass through (free thinking)
   - `len(output) >= budget` → force transition tokens one by one
   - EOS before `</think>` with guard enabled → force transition if EOS is
     argmax or `P(EOS)` exceeds threshold; otherwise mask EOS
   - `</think>` in output → pass through (thinking done)

**Hybrid model safety**: For Qwen3.5 with `enable_thinking=False`, the chat template injects `<think>\n\n</think>\n\n` into the prompt. The processor detects `</think>` in the prompt tokens via `_new_state()` and skips itself entirely — no interference with non-thinking responses regardless of length.

**Template-injected `<think>`**: For Qwen3.5 with `enable_thinking=True`, the chat template injects `<think>\n` into the prompt (not into generated output). The processor handles this by counting all output tokens as thinking tokens when `<think>` is not found in `output_ids`.
