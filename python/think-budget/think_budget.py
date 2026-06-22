"""vLLM logits processor for controlling thinking token budget.

Limits the number of tokens a model can generate inside <think>...</think>
blocks.  When the budget is reached, injects a transition instruction
(from Qwen3 paper) telling the model to wrap up and answer, then forces
</think>.

Compatible with any model using <think>...</think> reasoning tokens
(Qwen3-Thinking, Qwen3.5 hybrid, DeepSeek-R1, etc.).  Works both when
the model generates <think> itself (always-thinking models) and when
<think> is injected by the chat template (Qwen3.5 with enable_thinking).
Safe for hybrid reasoners — automatically detects when the template
disables thinking (</think> in prompt) and becomes a no-op.
Requires vLLM >= 0.11.0 (V1 engine).

Usage (vLLM CLI):
    THINK_BUDGET=1024 python -m vllm.entrypoints.openai.api_server \\
        --model Qwen/Qwen3-VL-8B-Thinking \\
        --reasoning-parser qwen3 \\
        --logits-processors think_budget:ThinkBudgetProcessor

    Set THINK_BUDGET=0 to disable budget enforcement (processor becomes no-op).

Environment variables:
    THINK_BUDGET          Default max thinking tokens (default: 1024, 0=disabled)
    THINK_END_TOKEN_ID    Override </think> token ID (auto-detected by default)
    THINK_EOS_TOKEN_ID    Override EOS token ID (auto-detected by default)
    THINK_TRANSITION      Override transition instruction text
    THINK_ENSURE_END_BEFORE_EOS
                          If true, prevent EOS before </think>; force transition
                          when EOS is argmax or above probability threshold
    THINK_EOS_PROB_THRESHOLD
                          Force transition before </think> when P(EOS) >= threshold
    THINK_BAN_EOS_AFTER_THINK_END_TOKENS
                          Ban EOS for this many answer tokens after forced </think>

Per-request override via vllm_xargs (OpenAI-compatible API):
    extra_body={"vllm_xargs": {"think_budget": 256}}
"""

__version__ = "0.4.0"

import logging
import os
from functools import partial
from typing import TYPE_CHECKING, Optional

import torch

from vllm.v1.sample.logits_processor import AdapterLogitsProcessor

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.sampling_params import SamplingParams

logger = logging.getLogger("think_budget")

# Default transition instruction from Qwen3 paper (Section 3.4, Thinking Budget)
_DEFAULT_TRANSITION = (
    "Considering the limited time by the user, I have to give the "
    "solution based on the thinking directly now.\n</think>\n\n"
)

_DEFAULT_THINK_END_ID = 151668


def _load_tokenizer(vllm_config: "VllmConfig"):
    """Load the model tokenizer for token ID detection and transition encoding."""
    try:
        from transformers import AutoTokenizer

        model = vllm_config.model_config.model
        return AutoTokenizer.from_pretrained(model, trust_remote_code=True)
    except Exception as exc:
        logger.warning("Could not load tokenizer: %s", exc)
        return None


def _detect_token_id(
    tokenizer, vllm_config: "VllmConfig", token: str, env_var: str, default: int
) -> int:
    """Auto-detect a special token ID. Priority: env var > tokenizer > default."""
    env_id = os.environ.get(env_var)
    if env_id:
        return int(env_id)

    if tokenizer is not None:
        try:
            token_id = tokenizer.convert_tokens_to_ids(token)
            if token_id is not None and token_id != getattr(
                tokenizer, "unk_token_id", None
            ):
                logger.info(
                    "Auto-detected %s token ID = %d from %s",
                    token,
                    token_id,
                    vllm_config.model_config.model,
                )
                return token_id
        except Exception as exc:
            logger.warning("Could not detect %s token: %s", token, exc)

    logger.info("Using default %s token ID = %d", token, default)
    return default


def _parse_bool(value, default: bool = False) -> bool:
    """Parse bool-like env/extra_arg values."""
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        return value != 0
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def _detect_eos_token_id(tokenizer, vllm_config: "VllmConfig") -> Optional[int]:
    """Auto-detect EOS token ID. Priority: env var > tokenizer > None."""
    env_id = os.environ.get("THINK_EOS_TOKEN_ID")
    if env_id:
        return int(env_id)

    if tokenizer is not None:
        eos_id = getattr(tokenizer, "eos_token_id", None)
        if isinstance(eos_id, list):
            eos_id = eos_id[0] if eos_id else None
        if eos_id is not None:
            logger.info(
                "Auto-detected EOS token ID = %d from %s",
                eos_id,
                vllm_config.model_config.model,
            )
            return int(eos_id)

        eos_token = getattr(tokenizer, "eos_token", None)
        if eos_token is not None:
            try:
                token_id = tokenizer.convert_tokens_to_ids(eos_token)
                if token_id is not None and token_id != getattr(
                    tokenizer, "unk_token_id", None
                ):
                    logger.info(
                        "Auto-detected EOS token ID = %d from eos_token=%r",
                        token_id,
                        eos_token,
                    )
                    return int(token_id)
            except Exception as exc:
                logger.warning("Could not detect EOS token: %s", exc)

    logger.warning(
        "Could not auto-detect EOS token ID; EOS-before-</think> guard "
        "will be disabled unless THINK_EOS_TOKEN_ID is set"
    )
    return None


def _encode_transition(tokenizer, think_end_id: int) -> list[int]:
    """Encode the transition instruction into token IDs.

    Falls back to just [think_end_id] if tokenizer is unavailable.
    """
    text = os.environ.get("THINK_TRANSITION") or _DEFAULT_TRANSITION

    if tokenizer is not None:
        try:
            ids = tokenizer.encode(text, add_special_tokens=False)
            if ids:
                logger.info(
                    "Transition instruction: %d tokens (%r...)",
                    len(ids),
                    text[:60],
                )
                return ids
        except Exception as exc:
            logger.warning("Could not encode transition: %s", exc)

    logger.info("Falling back to single </think> token for transition")
    return [think_end_id]


class ThinkBudgetProcessor(AdapterLogitsProcessor):
    """Server-level logits processor that limits thinking tokens.

    When a model is inside a <think> block and has generated more than
    ``THINK_BUDGET`` tokens, this processor forces a transition instruction
    (from Qwen3 paper) followed by ``</think>``, guiding the model to
    produce a coherent answer based on its accumulated reasoning.

    Configuration is via environment variables so that no custom kwargs
    are needed on the ``--logits-processors`` CLI (vLLM only passes
    ``(vllm_config, device, is_pin_memory)`` to server-level processors).
    """

    def __init__(
        self,
        vllm_config: "VllmConfig",
        device: torch.device,
        is_pin_memory: bool,
    ) -> None:
        super().__init__(vllm_config, device, is_pin_memory)
        self.max_think_tokens = int(os.environ.get("THINK_BUDGET", 1024))
        self.ensure_end_before_eos = _parse_bool(
            os.environ.get("THINK_ENSURE_END_BEFORE_EOS"), False
        )
        self.eos_prob_threshold = float(
            os.environ.get("THINK_EOS_PROB_THRESHOLD", 0.0)
        )
        self.ban_eos_after_think_end_tokens = int(
            os.environ.get("THINK_BAN_EOS_AFTER_THINK_END_TOKENS", 0)
        )

        tokenizer = _load_tokenizer(vllm_config)
        self.think_start_id = _detect_token_id(
            tokenizer, vllm_config, "<think>", "THINK_START_TOKEN_ID",
            _DEFAULT_THINK_END_ID - 1,  # <think> is typically one before </think>
        )
        self.think_end_id = _detect_token_id(
            tokenizer, vllm_config, "</think>", "THINK_END_TOKEN_ID",
            _DEFAULT_THINK_END_ID,
        )
        self.eos_token_id = _detect_eos_token_id(tokenizer, vllm_config)
        self.transition_ids = _encode_transition(tokenizer, self.think_end_id)

        if (
            self.max_think_tokens > 0
            or self.ensure_end_before_eos
            or self.ban_eos_after_think_end_tokens > 0
        ):
            logger.info(
                "ThinkBudgetProcessor active: budget=%d tokens, transition=%d "
                "tokens, <think> id=%d, </think> id=%d, eos id=%s, "
                "ensure_end_before_eos=%s, eos_prob_threshold=%.4f, "
                "ban_eos_after_think_end_tokens=%d",
                self.max_think_tokens,
                len(self.transition_ids),
                self.think_start_id,
                self.think_end_id,
                self.eos_token_id,
                self.ensure_end_before_eos,
                self.eos_prob_threshold,
                self.ban_eos_after_think_end_tokens,
            )
        else:
            logger.info(
                "ThinkBudgetProcessor disabled (THINK_BUDGET=0 and EOS guards off)"
            )

    def new_req_logits_processor(
        self, params: "SamplingParams"
    ) -> Optional["_ThinkBudgetReqProcessor"]:
        # Per-request budget via vllm_xargs; fall back to server default
        budget = self.max_think_tokens
        ensure_end_before_eos = self.ensure_end_before_eos
        eos_prob_threshold = self.eos_prob_threshold
        ban_eos_after_think_end_tokens = self.ban_eos_after_think_end_tokens
        if params.extra_args:
            budget = int(params.extra_args.get("think_budget", budget))
            ensure_end_before_eos = _parse_bool(
                params.extra_args.get(
                    "ensure_think_end_before_eos", ensure_end_before_eos
                ),
                ensure_end_before_eos,
            )
            eos_prob_threshold = float(
                params.extra_args.get(
                    "think_eos_prob_threshold", eos_prob_threshold
                )
            )
            ban_eos_after_think_end_tokens = int(
                params.extra_args.get(
                    "ban_eos_after_think_end_tokens",
                    ban_eos_after_think_end_tokens,
                )
            )
        if (
            budget <= 0
            and not ensure_end_before_eos
            and ban_eos_after_think_end_tokens <= 0
        ):
            return None
        return _ThinkBudgetReqProcessor(
            max_think_tokens=budget,
            think_start_id=self.think_start_id,
            think_end_id=self.think_end_id,
            eos_token_id=self.eos_token_id,
            transition_ids=self.transition_ids,
            ensure_end_before_eos=ensure_end_before_eos,
            eos_prob_threshold=eos_prob_threshold,
            ban_eos_after_think_end_tokens=ban_eos_after_think_end_tokens,
        )

    def _new_state(
        self,
        params: "SamplingParams",
        prompt_ids: Optional[list[int]],
        output_ids: list[int],
    ) -> Optional[partial[torch.Tensor]]:
        # If template disabled thinking (</think> already in prompt),
        # skip the processor entirely.  Qwen3.5 with enable_thinking=False
        # injects <think>\n\n</think>\n\n into the prompt.
        if prompt_ids and self.think_end_id in prompt_ids:
            return None
        return super()._new_state(params, prompt_ids, output_ids)

    def is_argmax_invariant(self) -> bool:
        return False


def make_think_budget_processor(
    tokenizer,
    budget: int = 1024,
    transition: str | None = None,
    ensure_end_before_eos: bool = False,
    eos_prob_threshold: float = 0.0,
    ban_eos_after_think_end_tokens: int = 0,
) -> "_ThinkBudgetReqProcessor":
    """Create a thinking budget processor for use with ``vllm.LLM``.

    Usage with offline inference::

        from vllm import LLM, SamplingParams
        from think_budget import make_think_budget_processor

        llm = LLM(model="Qwen/Qwen3.5-9B")
        tokenizer = llm.get_tokenizer()
        processor = make_think_budget_processor(tokenizer, budget=512)

        params = SamplingParams(
            temperature=0.7,
            max_tokens=4096,
            logits_processors=[processor],
        )

        conversation = tokenizer.apply_chat_template(
            messages, tools=tools, tokenize=False,
            add_generation_prompt=True, enable_thinking=True,
        )
        outputs = llm.generate([conversation], sampling_params=params)

    Args:
        tokenizer: HuggingFace tokenizer (from ``llm.get_tokenizer()``
            or ``AutoTokenizer.from_pretrained(...)``).
        budget: Max thinking tokens.  ``0`` disables (returns a no-op).
        transition: Transition instruction text.  ``None`` uses the
            default from the Qwen3 paper.

    Returns:
        A callable ``(output_ids, logits) -> logits`` suitable for
        ``SamplingParams(logits_processors=[...])``.
    """
    if (
        budget <= 0
        and not ensure_end_before_eos
        and ban_eos_after_think_end_tokens <= 0
    ):
        return lambda output_ids, logits: logits

    # Detect token IDs
    think_end_id = _DEFAULT_THINK_END_ID
    think_start_id = _DEFAULT_THINK_END_ID - 1
    eos_token_id = None
    try:
        eid = tokenizer.convert_tokens_to_ids("</think>")
        if eid is not None and eid != getattr(tokenizer, "unk_token_id", None):
            think_end_id = eid
        sid = tokenizer.convert_tokens_to_ids("<think>")
        if sid is not None and sid != getattr(tokenizer, "unk_token_id", None):
            think_start_id = sid
        eos_token_id = getattr(tokenizer, "eos_token_id", None)
        if isinstance(eos_token_id, list):
            eos_token_id = eos_token_id[0] if eos_token_id else None
    except Exception:
        pass

    # Encode transition
    text = transition if transition is not None else _DEFAULT_TRANSITION
    try:
        transition_ids = tokenizer.encode(text, add_special_tokens=False)
    except Exception:
        transition_ids = [think_end_id]

    return _ThinkBudgetReqProcessor(
        max_think_tokens=budget,
        think_start_id=think_start_id,
        think_end_id=think_end_id,
        eos_token_id=eos_token_id,
        transition_ids=transition_ids,
        ensure_end_before_eos=ensure_end_before_eos,
        eos_prob_threshold=eos_prob_threshold,
        ban_eos_after_think_end_tokens=ban_eos_after_think_end_tokens,
    )


class _ThinkBudgetReqProcessor:
    """Per-request thinking budget enforcement.

    Handles two cases:
    - ``<think>`` in output_ids: model generated it (always-thinking).
      Counts tokens from that position.
    - ``<think>`` NOT in output_ids: template injected it into the prompt
      (Qwen3.5 with ``enable_thinking=True``).  Counts all output tokens.

    Note: when ``enable_thinking=False``, the template injects both
    ``<think>`` and ``</think>`` into the prompt.  The server-level
    ``ThinkBudgetProcessor._new_state`` detects this and skips creating
    the per-request processor entirely.

    When the budget is reached, forces a transition instruction token by
    token, then ``</think>``.  The transition instruction (from Qwen3
    paper) tells the model to wrap up based on its partial reasoning:

        "Considering the limited time by the user, I have to give the
         solution based on the thinking directly now.\\n</think>\\n\\n"

    This produces much better answers than abruptly forcing ``</think>``.
    """

    def __init__(
        self,
        max_think_tokens: int,
        think_start_id: int,
        think_end_id: int,
        eos_token_id: Optional[int],
        transition_ids: list[int],
        ensure_end_before_eos: bool = False,
        eos_prob_threshold: float = 0.0,
        ban_eos_after_think_end_tokens: int = 0,
    ) -> None:
        self.max_think_tokens = max_think_tokens
        self.think_start_id = think_start_id
        self.think_end_id = think_end_id
        self.eos_token_id = eos_token_id
        self.transition_ids = transition_ids
        self.ensure_end_before_eos = ensure_end_before_eos
        self.eos_prob_threshold = eos_prob_threshold
        self.ban_eos_after_think_end_tokens = ban_eos_after_think_end_tokens
        self.transition_start_len: Optional[int] = None
        self.transition_done_len: Optional[int] = None

    def _force_token(self, logits: torch.Tensor, token_id: int) -> torch.Tensor:
        logits = torch.full_like(logits, float("-inf"))
        logits[token_id] = 1.0
        return logits

    def _force_transition(self, output_ids: list[int], logits: torch.Tensor) -> torch.Tensor:
        if self.transition_start_len is None:
            self.transition_start_len = len(output_ids)

        pos = len(output_ids) - self.transition_start_len
        if pos < len(self.transition_ids):
            forced_id = self.transition_ids[pos]
        else:
            # Safety fallback: force </think> if transition exhausted.
            forced_id = self.think_end_id

        return self._force_token(logits, forced_id)

    def _finish_transition_if_needed(self, output_ids: list[int]) -> None:
        if self.transition_start_len is None:
            return

        if len(output_ids) - self.transition_start_len >= len(self.transition_ids):
            self.transition_done_len = len(output_ids)
            self.transition_start_len = None

    def _ban_eos(self, logits: torch.Tensor) -> torch.Tensor:
        if self.eos_token_id is None:
            return logits
        logits = logits.clone()
        logits[self.eos_token_id] = float("-inf")
        return logits

    def _eos_pressure_triggered(self, logits: torch.Tensor) -> bool:
        if not self.ensure_end_before_eos or self.eos_token_id is None:
            return False

        eos_is_argmax = torch.argmax(logits).item() == self.eos_token_id
        if eos_is_argmax:
            return True

        if self.eos_prob_threshold > 0:
            eos_prob = torch.softmax(logits.float(), dim=-1)[self.eos_token_id]
            return eos_prob.item() >= self.eos_prob_threshold

        return False

    def _ban_eos_after_think_end(self, output_ids: list[int], logits: torch.Tensor) -> torch.Tensor:
        if (
            self.eos_token_id is None
            or self.ban_eos_after_think_end_tokens <= 0
            or self.think_end_id not in output_ids
        ):
            return logits

        if self.transition_done_len is not None:
            answer_start_len = self.transition_done_len
        else:
            answer_start_len = output_ids.index(self.think_end_id) + 1

        answer_tokens = len(output_ids) - answer_start_len
        if answer_tokens < self.ban_eos_after_think_end_tokens:
            return self._ban_eos(logits)

        return logits

    def __call__(
        self, output_ids: list[int], logits: torch.Tensor
    ) -> torch.Tensor:
        self._finish_transition_if_needed(output_ids)

        if self.transition_start_len is not None:
            return self._force_transition(output_ids, logits)

        # If </think> already generated, thinking is done — pass through
        if self.think_end_id in output_ids:
            return self._ban_eos_after_think_end(output_ids, logits)

        # Count tokens inside <think> block.
        # Case 1: <think> in output_ids → model generated it (always-thinking)
        # Case 2: <think> NOT in output_ids → template injected it into prompt,
        #         so all output tokens are thinking tokens (count from 0)
        if self.think_start_id in output_ids:
            think_start_pos = output_ids.index(self.think_start_id)
            think_tokens = len(output_ids) - think_start_pos - 1
        else:
            think_tokens = len(output_ids)

        # Budget not yet reached — let the model think freely
        budget_triggered = (
            self.max_think_tokens > 0
            and think_tokens >= self.max_think_tokens
        )
        eos_triggered = self._eos_pressure_triggered(logits)

        if budget_triggered or eos_triggered:
            return self._force_transition(output_ids, logits)

        # If enabled, EOS is never allowed before </think>.  If EOS pressure
        # was strong enough, the branch above already converted it into a
        # forced transition; otherwise this prevents sampled premature stops.
        if self.ensure_end_before_eos:
            return self._ban_eos(logits)

        return logits
