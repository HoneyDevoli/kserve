"""Test that thinking budget works per-request with Qwen3.5-9B.

Usage:
    python test_budget.py http://HOST:PORT/v1
    python test_budget.py                        # default: http://localhost:8150/v1
"""

import sys
from openai import OpenAI

BASE_URL = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8150/v1"
MODEL = "Qwen/Qwen3.5-9B"

client = OpenAI(base_url=BASE_URL, api_key="EMPTY")


def _get_reasoning(msg):
    """Extract reasoning content from response (vLLM puts it in 'reasoning' field)."""
    dump = msg.model_dump()
    return dump.get("reasoning") or dump.get("reasoning_content") or ""


def test_thinking(budget: int | None, label: str):
    """Send a request and report thinking vs answer token counts."""
    extra_body = {"chat_template_kwargs": {"enable_thinking": True}}
    if budget is not None:
        extra_body["vllm_xargs"] = {"think_budget": budget}

    print(f"\n{'='*60}")
    print(f"TEST: {label} (budget={budget})")
    print(f"{'='*60}")

    resp = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": "Explain step by step how to solve: what is 127 * 843?"}],
        extra_body=extra_body,
        max_tokens=4096,
        temperature=0.6,
    )

    msg = resp.choices[0].message
    reasoning = _get_reasoning(msg)
    content = msg.content or ""

    reasoning_tokens = None
    if resp.usage.completion_tokens_details:
        reasoning_tokens = getattr(resp.usage.completion_tokens_details, "reasoning_tokens", None)

    print(f"  Reasoning tokens: {reasoning_tokens if reasoning_tokens is not None else '?'}")
    print(f"  Reasoning chars:  {len(reasoning)}")
    print(f"  Answer chars:     {len(content)}")
    print(f"  Total completion: {resp.usage.completion_tokens} tokens")

    if budget is not None and budget > 0 and reasoning_tokens is not None:
        max_expected = budget + 30
        if reasoning_tokens <= max_expected:
            print(f"  OK: reasoning_tokens ({reasoning_tokens}) <= budget + 30 ({max_expected})")
        else:
            print(f"  FAIL: reasoning_tokens ({reasoning_tokens}) > budget + 30 ({max_expected})")

    print(f"  Reasoning preview: {reasoning[:200]}...")
    print(f"  Answer preview:    {content[:200]}...")

    return resp


def test_no_thinking():
    """Test that disable_thinking still works (processor is auto-skipped)."""
    print(f"\n{'='*60}")
    print(f"TEST: thinking disabled (should have no reasoning)")
    print(f"{'='*60}")

    resp = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": "What is 2+2?"}],
        extra_body={
            "chat_template_kwargs": {"enable_thinking": False},
        },
        max_tokens=256,
        temperature=0.6,
    )

    msg = resp.choices[0].message
    reasoning = _get_reasoning(msg)
    content = msg.content or ""

    print(f"  Reasoning: '{reasoning[:100]}'")
    print(f"  Answer:    '{content[:200]}'")
    print(f"  Total tokens: {resp.usage.completion_tokens}")
    has_reasoning = len(reasoning) > 0
    print(f"  OK: {'FAIL - has reasoning!' if has_reasoning else 'No reasoning (correct)'}")


if __name__ == "__main__":
    print("Testing think-budget with Qwen3.5-9B")
    print(f"Server: {BASE_URL}")

    # 1. Small budget — should limit thinking to ~64 + transition tokens
    test_thinking(budget=64, label="small budget (64)")

    # 2. Medium budget
    test_thinking(budget=256, label="medium budget (256)")

    # 3. Server default budget (1024)
    test_thinking(budget=None, label="server default (1024)")

    # 4. Budget=0 — disable processor, unlimited thinking
    test_thinking(budget=0, label="disabled (0) — unlimited")

    # 5. Thinking disabled entirely
    test_no_thinking()

    print(f"\n{'='*60}")
    print("ALL TESTS DONE")
    print(f"{'='*60}")
