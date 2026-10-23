"""GPU-only test: fully-masked rows → zero output, zero grads (no NaN).

Prompt 004 M-2. A query row whose block mask gates away every key
hits ``l_i == 0`` and ``m_i == -inf`` in the forward kernel. Before
the fix the stored LSE was ``-inf``; the backward then evaluated
``exp(-inf - -inf) = exp(NaN) = NaN`` and poisoned ``dQ, dK, dV``.
The forward now stores a finite sentinel for the LSE and nulls the
``m_i`` of empty rows so the backward produces p=0 → zero contribution.

A well-formed StreamingLLM mask with ``sink_size >= 1`` never has an
empty row, so this cannot fire for the configurations the benchmark
exercises. The GPU-only guard is in place because ``block_sparse_attention``
requires CUDA + Triton; CPU hosts skip this file.
"""

from __future__ import annotations

import pytest
import torch

triton = pytest.importorskip("triton")
if not torch.cuda.is_available():  # pragma: no cover — covered by GPU host
    pytest.skip("CUDA not available", allow_module_level=True)

from sparse_triton_kernel import block_sparse_attention  # noqa: E402


def test_fully_masked_row_outputs_zero_and_no_nan() -> None:
    """A row gated to zero blocks must produce finite zeros, not NaN."""
    B, H, N_Q, N_KV, D = 1, 2, 32, 32, 16
    bm = bn = 16

    device = torch.device("cuda")
    dtype = torch.float32
    q = torch.randn(B, H, N_Q, D, device=device, dtype=dtype, requires_grad=True)
    k = torch.randn(B, H, N_KV, D, device=device, dtype=dtype, requires_grad=True)
    v = torch.randn(B, H, N_KV, D, device=device, dtype=dtype, requires_grad=True)

    # Custom block mask: block (0, :) is fully masked (all zeros).  The
    # second query block (rows 16..31) still attends to the second KV
    # block, so dKdV has non-trivial work and the "empty row" path is
    # exercised in isolation on rows 0..15.
    n_qb = N_Q // bm
    n_kb = N_KV // bn
    block_mask = torch.zeros((n_qb, n_kb), dtype=torch.int32, device=device)
    block_mask[1, 1] = 1  # one kept pair for the second query block

    out = block_sparse_attention(
        q, k, v, block_mask,
        block_m=bm, block_n=bn,
        is_causal=False,  # custom mask; no streamingllm token mask
    )

    assert torch.isfinite(out).all(), "forward output has NaN or inf"
    # Rows 0..15 are fully masked → their output must be exactly zero.
    assert torch.equal(
        out[:, :, :bm, :], torch.zeros_like(out[:, :, :bm, :])
    ), "fully-masked rows did not produce zero output"

    grad_out = torch.randn_like(out)
    out.backward(grad_out)
    for name, g in (("dQ", q.grad), ("dK", k.grad), ("dV", v.grad)):
        assert g is not None, f"{name} is None"
        assert torch.isfinite(g).all(), f"{name} contains NaN or inf"
    # Rows 0..15 contributed nothing → their dQ rows must be zero.
    assert q.grad is not None
    assert torch.equal(
        q.grad[:, :, :bm, :], torch.zeros_like(q.grad[:, :, :bm, :])
    ), "dQ for fully-masked rows is not zero"
