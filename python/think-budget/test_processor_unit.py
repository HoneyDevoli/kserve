"""Unit tests for ThinkBudget request-level logits processing.

These tests exercise the processor directly without starting a vLLM server.
"""

import torch

from think_budget import _ThinkBudgetReqProcessor


def _processor(**kwargs):
    params = {
        "max_think_tokens": 0,
        "think_start_id": 3,
        "think_end_id": 4,
        "eos_token_id": 2,
        "transition_ids": [7, 4, 8],
        "ensure_end_before_eos": True,
        "eos_prob_threshold": 0.0,
        "ban_eos_after_think_end_tokens": 0,
    }
    params.update(kwargs)
    return _ThinkBudgetReqProcessor(**params)


def _logits(values):
    return torch.tensor(values, dtype=torch.float32)


def test_masks_eos_before_think_end_when_eos_is_not_pressure():
    proc = _processor()
    logits = _logits([0.0, 4.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

    out = proc(output_ids=[10, 11], logits=logits)

    assert torch.isneginf(out[2])
    assert out[1].item() == 4.0


def test_forces_transition_when_eos_is_argmax_before_think_end():
    proc = _processor()
    logits = _logits([0.0, 1.0, 5.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

    out = proc(output_ids=[10, 11], logits=logits)

    assert torch.argmax(out).item() == 7


def test_forces_transition_when_eos_probability_reaches_threshold():
    proc = _processor(eos_prob_threshold=0.2)
    logits = _logits([0.0, 3.0, 2.7, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

    out = proc(output_ids=[10, 11], logits=logits)

    assert torch.argmax(out).item() == 7


def test_forced_transition_continues_token_by_token():
    proc = _processor()
    logits = _logits([0.0, 1.0, 5.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

    first = proc(output_ids=[], logits=logits)
    second = proc(output_ids=[7], logits=logits)
    third = proc(output_ids=[7, 4], logits=logits)

    assert torch.argmax(first).item() == 7
    assert torch.argmax(second).item() == 4
    assert torch.argmax(third).item() == 8


def test_bans_eos_for_min_answer_tokens_after_forced_transition():
    proc = _processor(ban_eos_after_think_end_tokens=1)
    logits = _logits([0.0, 1.0, 5.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

    proc(output_ids=[], logits=logits)
    proc(output_ids=[7], logits=logits)
    proc(output_ids=[7, 4], logits=logits)
    out = proc(output_ids=[7, 4, 8], logits=logits)

    assert torch.isneginf(out[2])


def test_allows_eos_after_min_answer_tokens():
    proc = _processor(ban_eos_after_think_end_tokens=1)
    logits = _logits([0.0, 1.0, 5.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

    proc(output_ids=[], logits=logits)
    proc(output_ids=[7], logits=logits)
    proc(output_ids=[7, 4], logits=logits)
    proc(output_ids=[7, 4, 8], logits=logits)
    out = proc(output_ids=[7, 4, 8, 6], logits=logits)

    assert out[2].item() == 5.0
