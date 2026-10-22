"""CPU-runnable test that the oracle implements the StreamingLLM paper.

Prompt 001's H-2 finding: ``reference_block_sparse_attention`` lifted
the block mask to tokens, so it validated against a model of attention
that was not real StreamingLLM. Prompt 003 adds ``is_causal=True`` and
``(sink_size, window_size)`` kwargs that enforce the paper's token-level
mask inside every visited block.

This test runs on CPU (no Triton) and checks that, when invoked with
those kwargs and a well-formed block mask, the oracle's output equals
a direct dense-fp64 computation using the paper's token mask.
"""

from __future__ import annotations

import math

import pytest
import torch

from sparse_triton_kernel import (
    StreamingMaskSpec,
    make_streaming_mask,
    reference_block_sparse_attention,
    streaming_token_mask,
)


@pytest.mark.parametrize(
    "n_q,n_kv,sink,window,bm,bn",
    [
        (32, 32, 4, 8, 8, 8),
        (64, 64, 4, 16, 16, 16),
        (64, 128, 4, 32, 16, 32),
        (128, 128, 8, 32, 32, 32),
    ],
)
def test_oracle_matches_streamingllm_paper(
    n_q: int, n_kv: int, sink: int, window: int, bm: int, bn: int,
) -> None:
    torch.manual_seed(0)
    B, H, D = 1, 2, 16
    q = torch.randn(B, H, n_q, D, dtype=torch.float64)
    k = torch.randn(B, H, n_kv, D, dtype=torch.float64)
    v = torch.randn(B, H, n_kv, D, dtype=torch.float64)

    spec = StreamingMaskSpec(sink_size=sink, window_size=window)
    block_mask = make_streaming_mask(n_q, n_kv, spec, bm, bn, device="cpu")

    out = reference_block_sparse_attention(
        q, k, v, block_mask, block_m=bm, block_n=bn,
        is_causal=True, sink_size=sink, window_size=window,
    )

    # Direct dense computation using the paper's token mask only.
    scale = 1.0 / math.sqrt(D)
    scores = torch.matmul(q, k.transpose(-2, -1)) * scale
    token_mask = streaming_token_mask(n_q, n_kv, sink, window, device="cpu")
    scores = scores.masked_fill(~token_mask, float("-inf"))
    probs = torch.softmax(scores, dim=-1)
    probs = torch.nan_to_num(probs, nan=0.0)
    expected = torch.matmul(probs, v)

    assert torch.allclose(out, expected, atol=1e-12, rtol=1e-12), (
        f"Oracle output deviates from the paper's definition for "
        f"(n_q={n_q}, n_kv={n_kv}, sink={sink}, window={window}, "
        f"bm={bm}, bn={bn}). "
        f"max|Δ|={float((out - expected).abs().max()):.2e}"
    )


def test_oracle_pre_fix_leaks_the_bug() -> None:
    """Regression check: without ``is_causal``, the oracle keeps H-2's bug.

    If this test fails because the two outputs match, either someone
    accidentally changed the default of ``is_causal`` (which would hide
    the regression this prompt is here to fix), or the diagonal of the
    chosen shape happens to be fully-sink-covered. In either case,
    stop and investigate.
    """
    torch.manual_seed(0)
    B, H, D = 1, 1, 8
    n_q = n_kv = 32
    sink, window, bm, bn = 4, 8, 8, 8
    q = torch.randn(B, H, n_q, D, dtype=torch.float64)
    k = torch.randn(B, H, n_kv, D, dtype=torch.float64)
    v = torch.randn(B, H, n_kv, D, dtype=torch.float64)

    spec = StreamingMaskSpec(sink_size=sink, window_size=window)
    block_mask = make_streaming_mask(n_q, n_kv, spec, bm, bn, device="cpu")

    out_fixed = reference_block_sparse_attention(
        q, k, v, block_mask, block_m=bm, block_n=bn,
        is_causal=True, sink_size=sink, window_size=window,
    )
    out_pre_fix = reference_block_sparse_attention(
        q, k, v, block_mask, block_m=bm, block_n=bn,
        is_causal=False,
    )

    assert not torch.allclose(out_fixed, out_pre_fix, atol=1e-10), (
        "Expected the H-2 fix (is_causal=True + sink/window) to produce "
        "numerically different output from the pre-fix path. If this "
        "assertion fails, the kernel is probably still only applying "
        "block-level masking."
    )
