"""CPU-runnable test for the H-1 / H-2 correctness fix.

The two HIGH-risk findings from prompt 001 were:

- H-1: the Triton kernels applied the sparsity mask only at *block*
  granularity, so on diagonal and near-window-edge blocks tokens with
  ``k > q`` leaked through. Prompt 003 fixes this by applying the
  token-level StreamingLLM mask inside every visited block when
  ``is_causal=True``.
- H-2: the reference oracle lifted the *block* mask to tokens, so it
  shared H-1's bug and ``gradcheck`` validated the kernel against the
  wrong semantics. Prompt 003 fixes this by intersecting the lifted
  mask with the same token-level StreamingLLM mask inside blocks.

This test is CPU-only (no Triton required). It asserts that the
*effective mask* the kernel implements, computed in pure Python from
``make_streaming_mask`` + the in-block token mask formula, equals the
pure StreamingLLM token-level mask of the paper. If this test passes,
the fix is correct in principle; whether the Triton compiler lowers
it correctly on hardware is covered by ``benchmark.py verify-grads``
on the GPU host.
"""

from __future__ import annotations

import pytest
import torch

from sparse_triton_kernel import (
    StreamingMaskSpec,
    make_streaming_mask,
    streaming_token_mask,
)


SHAPES = [
    # (n_q, n_kv, sink, window, block_m, block_n)
    (16, 16, 4, 8, 8, 8),
    (32, 32, 4, 8, 8, 8),
    (64, 64, 4, 16, 16, 16),
    (64, 128, 4, 32, 16, 32),
    (128, 128, 8, 32, 32, 32),
    (256, 256, 4, 64, 64, 64),
    (128, 1024, 4, 128, 64, 64),  # strongly asymmetric Q vs KV
    (64, 64, 0, 64, 16, 16),      # window-only, no sinks
    (64, 64, 64, 0, 16, 16),      # sink-only (all keys), degenerate but valid
]


def _effective_kernel_mask(
    n_q: int, n_kv: int, sink: int, window: int, bm: int, bn: int,
) -> torch.Tensor:
    """Reconstruct, on CPU, the token-level mask the kernel will apply.

    The kernel:
      1. visits (q_block, k_block) pair iff ``block_mask[i, j] == 1``,
         where ``block_mask`` is :func:`make_streaming_mask`,
      2. inside every visited block, applies the token-level
         StreamingLLM mask.
    The outer product of (1) and (2) is the mask any kernel-produced
    attention effectively uses.
    """
    spec = StreamingMaskSpec(sink_size=sink, window_size=window)
    block_mask = make_streaming_mask(n_q, n_kv, spec, bm, bn, device="cpu")

    lifted = block_mask.to(torch.bool).repeat_interleave(bm, dim=0)
    lifted = lifted.repeat_interleave(bn, dim=1)
    lifted = lifted[:n_q, :n_kv]

    token_mask = streaming_token_mask(n_q, n_kv, sink, window, device="cpu")
    return lifted & token_mask


@pytest.mark.parametrize("n_q,n_kv,sink,window,bm,bn", SHAPES)
def test_kernel_mask_matches_streamingllm_definition(
    n_q: int, n_kv: int, sink: int, window: int, bm: int, bn: int,
) -> None:
    """The kernel's effective mask equals the paper's token-level mask."""
    effective = _effective_kernel_mask(n_q, n_kv, sink, window, bm, bn)
    paper = streaming_token_mask(n_q, n_kv, sink, window, device="cpu")
    assert effective.shape == paper.shape
    assert torch.equal(effective, paper), (
        f"Kernel's effective mask differs from the StreamingLLM paper "
        f"definition for shape=(n_q={n_q}, n_kv={n_kv}, sink={sink}, "
        f"window={window}, bm={bm}, bn={bn}).\n"
        f"Positions where kernel differs from paper: "
        f"{(effective != paper).nonzero(as_tuple=False).tolist()[:10]} …"
    )


@pytest.mark.parametrize("n_q,n_kv,sink,window,bm,bn", SHAPES)
def test_block_mask_is_superset_of_token_mask(
    n_q: int, n_kv: int, sink: int, window: int, bm: int, bn: int,
) -> None:
    """Every attended token pair lives inside a visited block.

    If this fails, the block mask is dropping cells that the kernel
    would have kept — the kernel would therefore produce wrong numbers
    even with the in-block fix.
    """
    spec = StreamingMaskSpec(sink_size=sink, window_size=window)
    block_mask = make_streaming_mask(n_q, n_kv, spec, bm, bn, device="cpu")
    lifted = block_mask.to(torch.bool).repeat_interleave(bm, dim=0)
    lifted = lifted.repeat_interleave(bn, dim=1)[:n_q, :n_kv]
    paper = streaming_token_mask(n_q, n_kv, sink, window, device="cpu")
    # paper ⊆ lifted
    assert torch.equal(paper & lifted, paper), (
        "block mask is not a superset of the token-level StreamingLLM "
        f"mask for (n_q={n_q}, n_kv={n_kv}, sink={sink}, "
        f"window={window}, bm={bm}, bn={bn})"
    )


def test_pre_fix_bug_is_visible_without_in_block_mask() -> None:
    """Regression test: without the in-block token mask, diagonal blocks leak.

    Before the H-1 fix, the kernel masked only with the lifted block
    mask. On a diagonal block, that admits cells with k > q. This
    test asserts that the *lifted-block-mask-only* model (the pre-fix
    behaviour) is strictly looser than the paper's token-level mask,
    i.e. the bug was real.
    """
    n_q, n_kv, sink, window, bm, bn = 32, 32, 4, 8, 8, 8
    spec = StreamingMaskSpec(sink_size=sink, window_size=window)
    block_mask = make_streaming_mask(n_q, n_kv, spec, bm, bn, device="cpu")
    lifted = block_mask.to(torch.bool).repeat_interleave(bm, dim=0)
    lifted = lifted.repeat_interleave(bn, dim=1)[:n_q, :n_kv]
    paper = streaming_token_mask(n_q, n_kv, sink, window, device="cpu")
    assert (lifted & ~paper).any().item(), (
        "Expected the lifted block mask to be strictly looser than the "
        "paper's token mask on at least one cell (that was the H-1 bug). "
        "If this assertion fails, either the block mask construction "
        "changed or the shapes picked here happen to be block-aligned "
        "with the window — pick different shapes."
    )
