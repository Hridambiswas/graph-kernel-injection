"""Independent, definition-based tests for the StreamingLLM mask.

Prompt 004 step 1.C — correctness fence against the repo's own helpers.

These tests implement the StreamingLLM paper's definition **by hand**
and check every mask producer against it:

    visible(i, j) = (j <= i) AND (j < S OR i - j < W)
    where i is the ABSOLUTE query position = (N_KV - N_Q) + row.

No repo helper is called inside the reference oracle, so a shared bug
in ``make_streaming_mask`` / ``streaming_token_mask`` / the oracle
cannot hide here (the H-2 class of problem from prompt 001).

Covers:
    - N_Q < S          (query length smaller than the sink count)
    - decode (N_Q=1)   where the visible count must be S+W when they
                        do not overlap, not ~S (prompt 004 bug B)
    - non-block-divisible N_KV (1000, 1023, 1025)
    - block mask is a SUPERSET of the token mask (no needed block
      silently skipped by the gate).
"""

from __future__ import annotations

import math
from typing import List, Tuple

import pytest
import torch

from sparse_triton_kernel import (
    StreamingMaskSpec,
    make_streaming_mask,
    reference_block_sparse_attention,
    streaming_token_mask,
)


# ---------------------------------------------------------------------------
# Hand-written reference: NO repo helper used inside this function.
# ---------------------------------------------------------------------------


def _visible(q_abs: int, k_abs: int, sink: int, window: int) -> bool:
    """Return True iff (q_abs, k_abs) is attended to under StreamingLLM."""
    if k_abs > q_abs:
        return False  # causal: k must be at most q
    if k_abs < sink:
        return True   # sink
    if q_abs - k_abs < window:
        return True   # window
    return False


def _reference_token_mask(
    n_q: int, n_kv: int, sink: int, window: int,
) -> torch.Tensor:
    """Build the [n_q, n_kv] token mask by hand, position by position."""
    q_offset = n_kv - n_q
    out = torch.zeros((n_q, n_kv), dtype=torch.bool)
    for row in range(n_q):
        q_abs = row + q_offset
        for k_abs in range(n_kv):
            if _visible(q_abs, k_abs, sink, window):
                out[row, k_abs] = True
    return out


# ---------------------------------------------------------------------------
# Shapes
# ---------------------------------------------------------------------------
# Named tuple for readability.  Each row is (n_q, n_kv, sink, window).


TOKEN_SHAPES: List[Tuple[int, int, int, int]] = [
    # decode step on a long KV cache (no sink/window overlap). The
    # single query sits at position N_KV - 1 and must see S + W keys.
    (1, 1000, 4, 256),
    # decode, window only
    (1, 1000, 0, 256),
    # decode, sink only
    (1, 1000, 4, 0),
    # decode, window covers everything
    (1, 1000, 4, 2048),
    # prefill-with-cache: N_Q == N_KV, no offset
    (16, 16, 4, 8),
    (64, 64, 4, 16),
    # tiny
    (2, 10, 4, 4),
    # non-block-divisible sequence lengths
    (1000, 1000, 4, 128),
    (1023, 1023, 4, 128),
    (1025, 1025, 4, 128),
    # N_Q smaller than sink size: queries are inside the sink prefix
    (4, 10, 8, 4),
    # asymmetric: decode-ish, N_Q > S but much smaller than N_KV
    (16, 1000, 4, 128),
    # the (128, 1024, ...) case used by the existing test
    (128, 1024, 4, 128),
]


# Shape × block-size matrix for the superset test. Includes partial
# edge blocks (prompt 004 M-1: mask builder uses ceiling division).
BLOCK_SHAPES: List[Tuple[int, int, int, int, int, int]] = [
    # (n_q, n_kv, sink, window, block_m, block_n)
    (1, 1000, 4, 256, 1, 64),      # decode non-divisible
    (1, 1023, 4, 256, 1, 64),      # decode non-divisible, edge
    (1, 1025, 4, 256, 1, 64),      # decode non-divisible, +1
    (16, 1000, 4, 128, 16, 64),    # asymmetric non-divisible
    (1000, 1000, 4, 128, 64, 64),  # prefill non-divisible
    (1023, 1023, 4, 128, 64, 64),  # prefill non-divisible, edge
    (1025, 1025, 4, 128, 64, 64),  # prefill non-divisible, +1
    (4, 10, 8, 4, 2, 2),           # N_Q < S, tiny blocks
    (16, 16, 4, 8, 8, 8),          # divisible, small
    (128, 1024, 4, 128, 16, 64),   # asymmetric divisible
]


# ---------------------------------------------------------------------------
# Tests: streaming_token_mask vs hand-written reference
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("n_q,n_kv,sink,window", TOKEN_SHAPES)
def test_streaming_token_mask_matches_hand_definition(
    n_q: int, n_kv: int, sink: int, window: int,
) -> None:
    """``streaming_token_mask`` must equal the by-hand paper definition."""
    repo = streaming_token_mask(n_q, n_kv, sink, window, device="cpu")
    hand = _reference_token_mask(n_q, n_kv, sink, window)
    assert repo.shape == hand.shape
    diff = (repo != hand).nonzero(as_tuple=False)
    assert diff.numel() == 0, (
        f"streaming_token_mask differs from the hand definition at "
        f"{diff[:10].tolist()} for (n_q={n_q}, n_kv={n_kv}, sink={sink}, "
        f"window={window}). Count-off: repo.sum={int(repo.sum())}, "
        f"hand.sum={int(hand.sum())}."
    )


def test_decode_sees_sink_plus_window_tokens() -> None:
    """Decode query at the end of the KV cache must see S + W keys.

    Bug B regression: if the kernel treated the lone decode query as
    position 0 instead of (N_KV - 1), it would see only ~S keys and
    fake a huge speedup.
    """
    n_q, n_kv, sink, window = 1, 1000, 4, 256
    assert sink + window <= n_kv - window + 1 + sink, (
        "test shape chosen so sink and window do NOT overlap"
    )
    mask = streaming_token_mask(n_q, n_kv, sink, window, device="cpu")
    visible = int(mask.sum().item())
    assert visible == sink + window, (
        f"Decode query must see sink+window ({sink + window}) keys, got "
        f"{visible}. Likely bug B regression (q_pos not bottom-right aligned)."
    )


def test_sink_does_not_leak_the_future() -> None:
    """A query with q < S must NOT attend to future sink keys k > q.

    Bug A regression: the old formula ``sink | (causal & window)``
    unconditionally included every k < sink_size, so q=0 could see
    k=1,2,3. Correct: ``causal & (sink | window)``.
    """
    n_q, n_kv, sink, window = 4, 4, 4, 1   # prefill, N_Q == N_KV
    mask = streaming_token_mask(n_q, n_kv, sink, window, device="cpu")
    # Row 0 can only see k=0 (itself); rows 1,2,3 see 0..row.
    for q in range(n_q):
        for k in range(n_kv):
            expected = k <= q
            assert bool(mask[q, k].item()) is expected, (
                f"Sink leak: mask[{q},{k}]={bool(mask[q, k].item())}, "
                f"expected {expected}."
            )


# ---------------------------------------------------------------------------
# Tests: make_streaming_mask is a SUPERSET of the token mask
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("n_q,n_kv,sink,window,bm,bn", BLOCK_SHAPES)
def test_block_mask_is_superset_of_hand_token_mask(
    n_q: int, n_kv: int, sink: int, window: int, bm: int, bn: int,
) -> None:
    """Every attended pair in the paper mask must live in a kept block.

    If this fails, the block gate is dropping a block that the kernel
    would have needed to visit — the output would be numerically wrong
    even with the correct in-block token mask.
    """
    spec = StreamingMaskSpec(sink_size=sink, window_size=window)
    block_mask = make_streaming_mask(n_q, n_kv, spec, bm, bn, device="cpu")

    # Expected block grid dimensions use ceiling division to match the
    # autograd bridge (prompt 004 M-1).
    expected_shape = (
        (n_q + bm - 1) // bm,
        (n_kv + bn - 1) // bn,
    )
    assert tuple(block_mask.shape) == expected_shape, (
        f"block mask shape {tuple(block_mask.shape)} != {expected_shape} "
        f"for (n_q={n_q}, n_kv={n_kv}, bm={bm}, bn={bn})"
    )

    lifted = block_mask.to(torch.bool).repeat_interleave(bm, dim=0)
    lifted = lifted.repeat_interleave(bn, dim=1)[:n_q, :n_kv]

    hand = _reference_token_mask(n_q, n_kv, sink, window)
    missing = hand & ~lifted
    assert missing.sum().item() == 0, (
        f"Block mask drops {int(missing.sum())} attended pair(s) that the "
        f"paper mask keeps, for (n_q={n_q}, n_kv={n_kv}, sink={sink}, "
        f"window={window}, bm={bm}, bn={bn}). Example: "
        f"{missing.nonzero(as_tuple=False)[:5].tolist()}"
    )


# ---------------------------------------------------------------------------
# Tests: reference_block_sparse_attention outputs the paper's mask
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "n_q,n_kv,sink,window,bm,bn",
    [
        (16, 16, 4, 8, 8, 8),
        (32, 32, 4, 8, 8, 8),
        (4, 32, 4, 8, 2, 8),      # decode-ish
        (1, 32, 4, 8, 1, 8),      # decode
        (16, 1000, 4, 128, 16, 64),  # asymmetric
    ],
)
def test_oracle_output_matches_hand_dense_computation(
    n_q: int, n_kv: int, sink: int, window: int, bm: int, bn: int,
) -> None:
    """The oracle's output must equal softmax(QK^T + hand_mask) @ V.

    Guards against both bug A (sink leak) and bug B (q_pos offset)
    silently flowing through the oracle too.
    """
    torch.manual_seed(0)
    B, H, D = 1, 2, 16
    q = torch.randn(B, H, n_q, D, dtype=torch.float64)
    k = torch.randn(B, H, n_kv, D, dtype=torch.float64)
    v = torch.randn(B, H, n_kv, D, dtype=torch.float64)

    spec = StreamingMaskSpec(sink_size=sink, window_size=window)
    block_mask = make_streaming_mask(n_q, n_kv, spec, bm, bn, device="cpu")

    repo_out = reference_block_sparse_attention(
        q, k, v, block_mask,
        block_m=bm, block_n=bn,
        is_causal=True, sink_size=sink, window_size=window,
    )

    scale = 1.0 / math.sqrt(D)
    scores = torch.matmul(q, k.transpose(-2, -1)) * scale
    hand = _reference_token_mask(n_q, n_kv, sink, window)
    scores = scores.masked_fill(~hand, float("-inf"))
    probs = torch.softmax(scores, dim=-1)
    probs = torch.nan_to_num(probs, nan=0.0)
    hand_out = torch.matmul(probs, v)

    assert torch.allclose(repo_out, hand_out, atol=1e-12, rtol=1e-12), (
        f"Oracle output differs from hand-built dense attention for "
        f"(n_q={n_q}, n_kv={n_kv}, sink={sink}, window={window}, "
        f"bm={bm}, bn={bn}). max|Δ|="
        f"{float((repo_out - hand_out).abs().max()):.2e}"
    )
