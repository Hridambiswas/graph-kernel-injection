"""Block-sparse Flash-Attention Triton kernel (forward + backward).

This module implements a hardware-aware, block-sparse variant of
Flash-Attention v2 targeting the long-context KV-cache regime
(:math:`N_{\\text{KV}} \\gg N_Q`).  The attention sparsity pattern is the
"StreamingLLM" mask (Xiao et al., 2024): a small set of **attention sink**
tokens at the start of the sequence together with a **sliding window**
over the most recent tokens.  Any user-supplied :math:`\\{0,1\\}`
block-mask is accepted — the StreamingLLM constructors here are provided
as a convenience.

Memory complexity (per head, per batch element)
-----------------------------------------------
Let :math:`N := N_{\\text{KV}}`, :math:`D` be the head dimension,
:math:`S` the sink length and :math:`W` the window length.

================  =================================  ===========================
Variant           HBM traffic (bytes read)           Peak workspace
================  =================================  ===========================
Eager (unfused)   :math:`\\Theta(N^2 + N D)`         :math:`\\Theta(N^2)`
Dense FlashAttn   :math:`\\Theta(N D)`               :math:`\\Theta(N D)`
**This kernel**   :math:`\\Theta((S + W) D)`         :math:`\\Theta(N D)`
================  =================================  ===========================

For a decode step with :math:`N = 10^5`, :math:`S + W = 2^{11}`, that is
a **≈ 49× reduction in KV HBM traffic** over dense FlashAttention and a
**≈ 2.4 × 10⁶× reduction** over the eager :math:`N^2` materialisation.

Design notes
------------
* **One kernel, two code paths.** The forward is a single Triton kernel
  with a compile-time ``IS_CAUSAL`` constexpr. The backward is split
  into (i) a tiny element-wise pre-processing kernel that materialises
  :math:`D_i = \\langle \\mathrm{d}O_i, O_i \\rangle` and (ii) two
  mirror-image kernels for :math:`\\partial Q` and
  :math:`(\\partial K, \\partial V)`. This matches FlashAttention v2
  (Dao, 2023) and sidesteps the atomic-add contention that a
  single-pass backward would incur on :math:`\\partial Q`.
* **Block-level sparsity.** The mask is :math:`[n_q, n_{kv}]` and is
  shared across the batch and head axes. The kernel reads one int32
  entry per inner-loop iteration before deciding whether to issue the
  bandwidth-heavy K/V loads.
* **SRAM layout.** The query tile ``Q`` and the running softmax state
  (``m_i``, ``l_i``, ``acc``) are register-resident across the KV
  reduction. Only ``K`` and ``V`` tiles stream through shared memory,
  the classic Flash-Attention trick.

References
----------
Dao, T.  *FlashAttention-2: Faster Attention with Better Parallelism and
Work Partitioning.*  arXiv:2307.08691, 2023.

Xiao, G. et al.  *Efficient Streaming Language Models with Attention
Sinks.*  ICLR 2024.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import torch

try:
    import triton
    import triton.language as tl

    _TRITON_AVAILABLE = True
except ImportError:  # pragma: no cover — Triton is Linux/CUDA-only
    triton = None  # type: ignore[assignment]
    tl = None  # type: ignore[assignment]
    _TRITON_AVAILABLE = False


# ---------------------------------------------------------------------------
# Mask construction utilities
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class StreamingMaskSpec:
    """Specification of a StreamingLLM-style sparsity pattern.

    Attributes
    ----------
    sink_size:
        Number of attention sink tokens at the start of the key/value
        sequence that **every** query position attends to. Typically 4
        in the StreamingLLM paper.
    window_size:
        Width of the causal sliding window: a query at position
        :math:`q` attends to keys in :math:`[q - W + 1,\\, q]`.
    """

    sink_size: int
    window_size: int

    def __post_init__(self) -> None:
        if self.sink_size < 0 or self.window_size < 0:
            raise ValueError("sink_size and window_size must be non-negative")


def make_streaming_mask(
    n_q: int,
    n_kv: int,
    spec: StreamingMaskSpec,
    block_m: int,
    block_n: int,
    device: torch.device | str = "cuda",
) -> torch.Tensor:
    """Build the block-level StreamingLLM sparsity mask.

    A block pair :math:`(i, j)` is kept iff some query token in query
    block :math:`i` attends to some key token in key block :math:`j`
    under the token-level StreamingLLM pattern (sink ∪ causal window).

    Parameters
    ----------
    n_q, n_kv:
        Token-level query and key sequence lengths. Must satisfy
        ``n_q % block_m == 0`` and ``n_kv % block_n == 0``.
    spec:
        The StreamingLLM parameters (sink + window).
    block_m, block_n:
        Row- and column-wise block sizes used by the Triton kernel.
    device:
        Device of the returned tensor.

    Returns
    -------
    torch.Tensor
        An ``int32`` tensor of shape ``[n_q // block_m, n_kv // block_n]``
        whose entries are 1 for "attend" and 0 for "skip".

    Complexity
    ----------
    :math:`O(n_q n_{kv} / (B_M B_N))` ops; no materialisation of the
    :math:`[n_q, n_{kv}]` dense mask.
    """
    if n_q % block_m != 0:
        raise ValueError(f"n_q ({n_q}) must be a multiple of block_m ({block_m})")
    if n_kv % block_n != 0:
        raise ValueError(f"n_kv ({n_kv}) must be a multiple of block_n ({block_n})")

    n_qb = n_q // block_m
    n_kb = n_kv // block_n

    # Compute the half-open token range covered by every block.
    q_lo = torch.arange(n_qb, device=device) * block_m
    q_hi = q_lo + block_m  # exclusive
    k_lo = torch.arange(n_kb, device=device) * block_n
    k_hi = k_lo + block_n

    # Sink: a key block j is a sink block iff k_lo[j] < sink_size.
    sink = (k_lo[None, :] < spec.sink_size).expand(n_qb, n_kb)

    # Causal window: block j overlaps the window of some q in block i iff
    #     k_lo[j]            <=   q_hi[i] - 1               (k <= q)
    #     k_hi[j] - 1        >=   q_lo[i] - window_size + 1 (k >= q - W + 1)
    causal_upper = k_lo[None, :] <= (q_hi[:, None] - 1)
    causal_lower = (k_hi[None, :] - 1) >= (q_lo[:, None] - spec.window_size + 1)
    window = causal_upper & causal_lower

    mask = (sink | window).to(torch.int32)
    return mask.contiguous()


def make_dense_mask(
    n_q: int,
    n_kv: int,
    block_m: int,
    block_n: int,
    causal: bool = False,
    device: torch.device | str = "cuda",
) -> torch.Tensor:
    """Build a fully-dense (or dense-causal) block mask.

    Used to validate numerical equivalence between this block-sparse
    kernel and vanilla (dense) attention under
    :func:`torch.nn.functional.scaled_dot_product_attention`.
    """
    if n_q % block_m != 0 or n_kv % block_n != 0:
        raise ValueError("block_m / block_n must divide n_q / n_kv")

    n_qb = n_q // block_m
    n_kb = n_kv // block_n

    if not causal:
        return torch.ones((n_qb, n_kb), dtype=torch.int32, device=device)

    q_hi = (torch.arange(n_qb, device=device) + 1) * block_m
    k_lo = torch.arange(n_kb, device=device) * block_n
    mask = k_lo[None, :] < q_hi[:, None]
    return mask.to(torch.int32).contiguous()


# ---------------------------------------------------------------------------
# Triton kernels
# ---------------------------------------------------------------------------
#
# The three kernels below (`_fwd_kernel`, `_bwd_dkdv_kernel`,
# `_bwd_dq_kernel`) plus the pre-processing kernel `_bwd_preprocess_kernel`
# constitute the full forward+backward.  All of them share the same
# strided-pointer convention: a tensor ``X`` of logical shape
# ``[B, H, N, D]`` is accessed by first offsetting the base pointer by
# ``off_z * stride_z + off_h * stride_h``, then issuing a
# ``make_block_ptr`` over the remaining ``[N, D]`` sub-tensor.  This
# matches the layout emitted by ``.contiguous()`` on a tensor permuted
# into ``BHND`` order.
# ---------------------------------------------------------------------------


if _TRITON_AVAILABLE:

    @triton.jit
    def _fwd_kernel(  # noqa: PLR0913 — mirroring Triton tutorial style
        Q, K, V, Out, Lse,
        BlockMask,
        softmax_scale,
        sink_size, window_size,
        stride_qz, stride_qh, stride_qm, stride_qd,
        stride_kz, stride_kh, stride_kn, stride_kd,
        stride_vz, stride_vh, stride_vn, stride_vd,
        stride_oz, stride_oh, stride_om, stride_od,
        stride_lz, stride_lh, stride_lm,
        stride_mq, stride_mk,
        Z, H, N_Q, N_KV,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        HEAD_DIM: tl.constexpr,
        NUM_KV_BLOCKS: tl.constexpr,
        IS_CAUSAL: tl.constexpr,
    ):
        """Block-sparse Flash-Attention forward kernel.

        Grid: ``(num_q_blocks, B * H)``.
        Each program instance owns one :math:`[B_M, D]` output tile.
        """
        pid_m = tl.program_id(axis=0)
        pid_bh = tl.program_id(axis=1)
        off_z = pid_bh // H
        off_h = pid_bh % H

        # ---- base-pointer offsets for this (batch, head) slice ----
        q_base = Q + off_z * stride_qz + off_h * stride_qh
        k_base = K + off_z * stride_kz + off_h * stride_kh
        v_base = V + off_z * stride_vz + off_h * stride_vh
        o_base = Out + off_z * stride_oz + off_h * stride_oh
        lse_base = Lse + off_z * stride_lz + off_h * stride_lh

        # ---- row/col index vectors (needed for bounds-safe loads) ----
        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_d = tl.arange(0, HEAD_DIM)

        # ---- load Q tile once into SRAM; it is reused for every KV block ----
        q_ptrs = q_base + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qd
        q_mask = offs_m[:, None] < N_Q
        q = tl.load(q_ptrs, mask=q_mask, other=0.0)

        # ---- running softmax state (fp32 for numerical stability) ----
        m_i = tl.full((BLOCK_M,), float("-inf"), dtype=tl.float32)
        l_i = tl.zeros((BLOCK_M,), dtype=tl.float32)
        acc = tl.zeros((BLOCK_M, HEAD_DIM), dtype=tl.float32)

        # ---- outer loop over KV blocks, gated by the sparsity mask ----
        offs_n = tl.arange(0, BLOCK_N)
        for kv_idx in range(NUM_KV_BLOCKS):
            mask_val = tl.load(BlockMask + pid_m * stride_mq + kv_idx * stride_mk)
            if mask_val != 0:
                kv_off = kv_idx * BLOCK_N
                k_ptrs = (
                    k_base
                    + (kv_off + offs_n)[None, :] * stride_kn
                    + offs_d[:, None] * stride_kd
                )
                v_ptrs = (
                    v_base
                    + (kv_off + offs_n)[:, None] * stride_vn
                    + offs_d[None, :] * stride_vd
                )
                kv_mask = (kv_off + offs_n) < N_KV
                k = tl.load(k_ptrs, mask=kv_mask[None, :], other=0.0)
                v = tl.load(v_ptrs, mask=kv_mask[:, None], other=0.0)

                # --- QK^T and (optional) in-block token-level mask ---
                # Block-level sparsity selects *which* (q_block, k_block)
                # pairs are visited; inside a visited pair we still have
                # to apply the token-level StreamingLLM mask, otherwise
                # diagonal / near-window-edge blocks let tokens see the
                # future (fix H-1 from docs/architecture_notes.md).
                qk = tl.dot(q, k) * softmax_scale  # [BLOCK_M, BLOCK_N]
                k_pos = kv_off + offs_n           # [BLOCK_N]
                if IS_CAUSAL:
                    sink = k_pos < sink_size                                   # [BLOCK_N]
                    causal = k_pos[None, :] <= offs_m[:, None]                 # [M, N]
                    windowed = k_pos[None, :] >= (offs_m[:, None] - window_size + 1)
                    token_mask = sink[None, :] | (causal & windowed)
                    qk = tl.where(kv_mask[None, :] & token_mask, qk, float("-inf"))
                else:
                    qk = tl.where(kv_mask[None, :], qk, float("-inf"))

                # Online softmax (FlashAttention-2 numerics):
                #   m_new = max(m_old, max_j(qk))
                #   p     = exp(qk - m_new)
                #   l_new = exp(m_old - m_new) * l_old + sum(p)
                #   acc   = exp(m_old - m_new) * acc + p @ v
                m_ij = tl.maximum(m_i, tl.max(qk, axis=1))
                alpha = tl.exp(m_i - m_ij)
                p = tl.exp(qk - m_ij[:, None])
                l_i = l_i * alpha + tl.sum(p, axis=1)
                acc = acc * alpha[:, None] + tl.dot(p.to(v.dtype), v)
                m_i = m_ij

        # ---- epilogue: finalise softmax division and store ----
        # If a query block is fully masked out (should not happen for a
        # well-formed StreamingLLM mask) we would hit `l_i == 0`; guard
        # against the division by zero.
        safe_l = tl.where(l_i > 0.0, l_i, 1.0)
        acc = acc / safe_l[:, None]

        o_ptrs = o_base + offs_m[:, None] * stride_om + offs_d[None, :] * stride_od
        tl.store(o_ptrs, acc.to(Out.dtype.element_ty), mask=q_mask)

        # Log-sum-exp is cached for the backward pass.
        lse_ptrs = lse_base + offs_m * stride_lm
        lse = m_i + tl.log(safe_l)
        tl.store(lse_ptrs, lse, mask=offs_m < N_Q)

    @triton.jit
    def _bwd_preprocess_kernel(
        Out, DO, Delta,
        stride_oz, stride_oh, stride_om, stride_od,
        stride_doz, stride_doh, stride_dom, stride_dod,
        stride_dz, stride_dh, stride_dm,
        Z, H, N_Q,
        BLOCK_M: tl.constexpr,
        HEAD_DIM: tl.constexpr,
    ):
        """Row-wise :math:`D_i = \\langle \\mathrm{d}O_i, O_i \\rangle`.

        This is the one piece of pre-work that FlashAttention v2 extracts
        from the main backward kernel so the inner loop can reuse ``D_i``
        without redundant reductions.
        """
        pid_m = tl.program_id(0)
        pid_bh = tl.program_id(1)
        off_z = pid_bh // H
        off_h = pid_bh % H

        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_d = tl.arange(0, HEAD_DIM)
        mask_m = offs_m < N_Q

        o_ptrs = (
            Out + off_z * stride_oz + off_h * stride_oh
            + offs_m[:, None] * stride_om + offs_d[None, :] * stride_od
        )
        do_ptrs = (
            DO + off_z * stride_doz + off_h * stride_doh
            + offs_m[:, None] * stride_dom + offs_d[None, :] * stride_dod
        )
        o = tl.load(o_ptrs, mask=mask_m[:, None], other=0.0).to(tl.float32)
        do = tl.load(do_ptrs, mask=mask_m[:, None], other=0.0).to(tl.float32)
        delta = tl.sum(o * do, axis=1)

        d_ptrs = Delta + off_z * stride_dz + off_h * stride_dh + offs_m * stride_dm
        tl.store(d_ptrs, delta, mask=mask_m)

    @triton.jit
    def _bwd_dkdv_kernel(  # noqa: PLR0913
        Q, K, V, DO, DK, DV, Lse, Delta,
        BlockMask,
        softmax_scale,
        sink_size, window_size,
        stride_qz, stride_qh, stride_qm, stride_qd,
        stride_kz, stride_kh, stride_kn, stride_kd,
        stride_vz, stride_vh, stride_vn, stride_vd,
        stride_doz, stride_doh, stride_dom, stride_dod,
        stride_dkz, stride_dkh, stride_dkn, stride_dkd,
        stride_dvz, stride_dvh, stride_dvn, stride_dvd,
        stride_lz, stride_lh, stride_lm,
        stride_dltz, stride_dlth, stride_dltm,
        stride_mq, stride_mk,
        Z, H, N_Q, N_KV,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        HEAD_DIM: tl.constexpr,
        NUM_Q_BLOCKS: tl.constexpr,
        IS_CAUSAL: tl.constexpr,
    ):
        """Backward pass computing :math:`\\partial K`, :math:`\\partial V`.

        Grid: ``(num_kv_blocks, B * H)``. The outer axis is **KV**: this
        kernel owns the entire reduction over Q blocks for one K/V tile,
        which lets us keep ``dK, dV`` register-resident and emit a single
        contiguous store at the end — no atomic adds.
        """
        pid_n = tl.program_id(0)
        pid_bh = tl.program_id(1)
        off_z = pid_bh // H
        off_h = pid_bh % H

        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        offs_d = tl.arange(0, HEAD_DIM)
        mask_n = offs_n < N_KV

        k_base = K + off_z * stride_kz + off_h * stride_kh
        v_base = V + off_z * stride_vz + off_h * stride_vh
        q_base = Q + off_z * stride_qz + off_h * stride_qh
        do_base = DO + off_z * stride_doz + off_h * stride_doh
        lse_base = Lse + off_z * stride_lz + off_h * stride_lh
        delta_base = Delta + off_z * stride_dltz + off_h * stride_dlth

        k_ptrs = k_base + offs_n[:, None] * stride_kn + offs_d[None, :] * stride_kd
        v_ptrs = v_base + offs_n[:, None] * stride_vn + offs_d[None, :] * stride_vd
        k = tl.load(k_ptrs, mask=mask_n[:, None], other=0.0)
        v = tl.load(v_ptrs, mask=mask_n[:, None], other=0.0)

        dk = tl.zeros((BLOCK_N, HEAD_DIM), dtype=tl.float32)
        dv = tl.zeros((BLOCK_N, HEAD_DIM), dtype=tl.float32)

        offs_m = tl.arange(0, BLOCK_M)
        for qb in range(NUM_Q_BLOCKS):
            mask_val = tl.load(BlockMask + qb * stride_mq + pid_n * stride_mk)
            if mask_val != 0:
                q_off = qb * BLOCK_M
                qm = (q_off + offs_m) < N_Q
                q_ptrs = (
                    q_base
                    + (q_off + offs_m)[:, None] * stride_qm
                    + offs_d[None, :] * stride_qd
                )
                do_ptrs = (
                    do_base
                    + (q_off + offs_m)[:, None] * stride_dom
                    + offs_d[None, :] * stride_dod
                )
                q = tl.load(q_ptrs, mask=qm[:, None], other=0.0)
                do = tl.load(do_ptrs, mask=qm[:, None], other=0.0)
                lse = tl.load(lse_base + (q_off + offs_m) * stride_lm, mask=qm, other=0.0)
                delta = tl.load(
                    delta_base + (q_off + offs_m) * stride_dltm, mask=qm, other=0.0
                )

                # Recompute P (not stashed) exactly as the forward did —
                # must apply the identical in-block token mask or the
                # bwd will leak gradients through forbidden pairs.
                qk = tl.dot(q, tl.trans(k)) * softmax_scale
                q_pos = q_off + offs_m   # [BLOCK_M] — absolute q position
                # offs_n here already equals (pid_n * BLOCK_N + arange),
                # i.e. absolute k position, so no further offset is needed.
                if IS_CAUSAL:
                    sink = offs_n < sink_size
                    causal = offs_n[None, :] <= q_pos[:, None]
                    windowed = offs_n[None, :] >= (q_pos[:, None] - window_size + 1)
                    token_mask = sink[None, :] | (causal & windowed)
                    qk = tl.where(qm[:, None] & mask_n[None, :] & token_mask, qk, float("-inf"))
                else:
                    qk = tl.where(qm[:, None] & mask_n[None, :], qk, float("-inf"))
                p = tl.exp(qk - lse[:, None])

                # dV += P^T @ dO  (reduce over Q within this block)
                dv += tl.dot(tl.trans(p).to(do.dtype), do)

                # dP = dO @ V^T   →   dS = P * (dP - D_i) * scale
                dp = tl.dot(do, tl.trans(v))
                ds = p * (dp - delta[:, None]) * softmax_scale

                # dK += dS^T @ Q
                dk += tl.dot(tl.trans(ds).to(q.dtype), q)

        dk_ptrs = (
            DK + off_z * stride_dkz + off_h * stride_dkh
            + offs_n[:, None] * stride_dkn + offs_d[None, :] * stride_dkd
        )
        dv_ptrs = (
            DV + off_z * stride_dvz + off_h * stride_dvh
            + offs_n[:, None] * stride_dvn + offs_d[None, :] * stride_dvd
        )
        tl.store(dk_ptrs, dk.to(DK.dtype.element_ty), mask=mask_n[:, None])
        tl.store(dv_ptrs, dv.to(DV.dtype.element_ty), mask=mask_n[:, None])

    @triton.jit
    def _bwd_dq_kernel(  # noqa: PLR0913
        Q, K, V, DO, DQ, Lse, Delta,
        BlockMask,
        softmax_scale,
        sink_size, window_size,
        stride_qz, stride_qh, stride_qm, stride_qd,
        stride_kz, stride_kh, stride_kn, stride_kd,
        stride_vz, stride_vh, stride_vn, stride_vd,
        stride_doz, stride_doh, stride_dom, stride_dod,
        stride_dqz, stride_dqh, stride_dqm, stride_dqd,
        stride_lz, stride_lh, stride_lm,
        stride_dltz, stride_dlth, stride_dltm,
        stride_mq, stride_mk,
        Z, H, N_Q, N_KV,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        HEAD_DIM: tl.constexpr,
        NUM_KV_BLOCKS: tl.constexpr,
        IS_CAUSAL: tl.constexpr,
    ):
        """Backward pass computing :math:`\\partial Q`.

        Grid: ``(num_q_blocks, B * H)``.  Mirror image of
        :func:`_bwd_dkdv_kernel` with the loop axes swapped.
        """
        pid_m = tl.program_id(0)
        pid_bh = tl.program_id(1)
        off_z = pid_bh // H
        off_h = pid_bh % H

        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_d = tl.arange(0, HEAD_DIM)
        mask_m = offs_m < N_Q

        q_base = Q + off_z * stride_qz + off_h * stride_qh
        k_base = K + off_z * stride_kz + off_h * stride_kh
        v_base = V + off_z * stride_vz + off_h * stride_vh
        do_base = DO + off_z * stride_doz + off_h * stride_doh
        lse_base = Lse + off_z * stride_lz + off_h * stride_lh
        delta_base = Delta + off_z * stride_dltz + off_h * stride_dlth

        q_ptrs = q_base + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qd
        do_ptrs = do_base + offs_m[:, None] * stride_dom + offs_d[None, :] * stride_dod
        q = tl.load(q_ptrs, mask=mask_m[:, None], other=0.0)
        do = tl.load(do_ptrs, mask=mask_m[:, None], other=0.0)
        lse = tl.load(lse_base + offs_m * stride_lm, mask=mask_m, other=0.0)
        delta = tl.load(delta_base + offs_m * stride_dltm, mask=mask_m, other=0.0)

        dq = tl.zeros((BLOCK_M, HEAD_DIM), dtype=tl.float32)

        offs_n = tl.arange(0, BLOCK_N)
        for kb in range(NUM_KV_BLOCKS):
            mask_val = tl.load(BlockMask + pid_m * stride_mq + kb * stride_mk)
            if mask_val != 0:
                kv_off = kb * BLOCK_N
                kvn = (kv_off + offs_n) < N_KV
                k_ptrs = (
                    k_base
                    + (kv_off + offs_n)[:, None] * stride_kn
                    + offs_d[None, :] * stride_kd
                )
                v_ptrs = (
                    v_base
                    + (kv_off + offs_n)[:, None] * stride_vn
                    + offs_d[None, :] * stride_vd
                )
                k = tl.load(k_ptrs, mask=kvn[:, None], other=0.0)
                v = tl.load(v_ptrs, mask=kvn[:, None], other=0.0)

                qk = tl.dot(q, tl.trans(k)) * softmax_scale
                k_pos = kv_off + offs_n   # [BLOCK_N] — absolute k position
                if IS_CAUSAL:
                    sink = k_pos < sink_size
                    causal = k_pos[None, :] <= offs_m[:, None]
                    windowed = k_pos[None, :] >= (offs_m[:, None] - window_size + 1)
                    token_mask = sink[None, :] | (causal & windowed)
                    qk = tl.where(mask_m[:, None] & kvn[None, :] & token_mask, qk, float("-inf"))
                else:
                    qk = tl.where(mask_m[:, None] & kvn[None, :], qk, float("-inf"))
                p = tl.exp(qk - lse[:, None])
                dp = tl.dot(do, tl.trans(v))
                ds = p * (dp - delta[:, None]) * softmax_scale
                dq += tl.dot(ds.to(k.dtype), k)

        dq_ptrs = (
            DQ + off_z * stride_dqz + off_h * stride_dqh
            + offs_m[:, None] * stride_dqm + offs_d[None, :] * stride_dqd
        )
        tl.store(dq_ptrs, dq.to(DQ.dtype.element_ty), mask=mask_m[:, None])


# ---------------------------------------------------------------------------
# `torch.autograd.Function` wrapper
# ---------------------------------------------------------------------------


class _BlockSparseFlashAttention(torch.autograd.Function):
    """Autograd bridge between PyTorch and the Triton kernels."""

    @staticmethod
    def forward(  # type: ignore[override]
        ctx,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        block_mask: torch.Tensor,
        softmax_scale: float,
        block_m: int,
        block_n: int,
        is_causal: bool,
        sink_size: int,
        window_size: int,
    ) -> torch.Tensor:
        if not _TRITON_AVAILABLE:
            raise RuntimeError(
                "Triton is unavailable. This kernel requires a CUDA GPU and "
                "Linux (Triton does not ship for macOS)."
            )
        if q.dim() != 4 or k.dim() != 4 or v.dim() != 4:
            raise ValueError("Q/K/V must be 4-D [B, H, N, D] tensors")
        if not (q.is_cuda and k.is_cuda and v.is_cuda):
            raise ValueError("All inputs must live on a CUDA device")

        B, H, N_Q, D = q.shape
        _, _, N_KV, _ = k.shape
        if k.shape != v.shape:
            raise ValueError("K and V must share shape")
        if k.shape[:2] != q.shape[:2] or k.shape[-1] != D:
            raise ValueError("Q/K/V batch/head/dim mismatch")

        n_qb = (N_Q + block_m - 1) // block_m
        n_kb = (N_KV + block_n - 1) // block_n
        if block_mask.shape != (n_qb, n_kb):
            raise ValueError(
                f"block_mask has shape {tuple(block_mask.shape)}, "
                f"expected {(n_qb, n_kb)}"
            )
        block_mask = block_mask.to(torch.int32).contiguous()

        o = torch.empty_like(q)
        lse = torch.empty((B, H, N_Q), dtype=torch.float32, device=q.device)

        grid = (n_qb, B * H)
        _fwd_kernel[grid](
            q, k, v, o, lse,
            block_mask,
            softmax_scale,
            int(sink_size), int(window_size),
            q.stride(0), q.stride(1), q.stride(2), q.stride(3),
            k.stride(0), k.stride(1), k.stride(2), k.stride(3),
            v.stride(0), v.stride(1), v.stride(2), v.stride(3),
            o.stride(0), o.stride(1), o.stride(2), o.stride(3),
            lse.stride(0), lse.stride(1), lse.stride(2),
            block_mask.stride(0), block_mask.stride(1),
            B, H, N_Q, N_KV,
            BLOCK_M=block_m,
            BLOCK_N=block_n,
            HEAD_DIM=D,
            NUM_KV_BLOCKS=n_kb,
            IS_CAUSAL=bool(is_causal),
            num_warps=4,
            num_stages=2,
        )

        ctx.save_for_backward(q, k, v, o, lse, block_mask)
        ctx.softmax_scale = softmax_scale
        ctx.block_m = block_m
        ctx.block_n = block_n
        ctx.is_causal = bool(is_causal)
        ctx.sink_size = int(sink_size)
        ctx.window_size = int(window_size)
        return o

    @staticmethod
    def backward(  # type: ignore[override]
        ctx, do: torch.Tensor
    ) -> Tuple[Optional[torch.Tensor], ...]:
        q, k, v, o, lse, block_mask = ctx.saved_tensors
        softmax_scale: float = ctx.softmax_scale
        block_m: int = ctx.block_m
        block_n: int = ctx.block_n
        is_causal: bool = ctx.is_causal
        sink_size: int = ctx.sink_size
        window_size: int = ctx.window_size

        do = do.contiguous()
        B, H, N_Q, D = q.shape
        _, _, N_KV, _ = k.shape
        n_qb = (N_Q + block_m - 1) // block_m
        n_kb = (N_KV + block_n - 1) // block_n

        dq = torch.zeros_like(q)
        dk = torch.zeros_like(k)
        dv = torch.zeros_like(v)
        delta = torch.empty((B, H, N_Q), dtype=torch.float32, device=q.device)

        # ---- 1. Pre-compute D_i = <dO_i, O_i> ----
        _bwd_preprocess_kernel[(n_qb, B * H)](
            o, do, delta,
            o.stride(0), o.stride(1), o.stride(2), o.stride(3),
            do.stride(0), do.stride(1), do.stride(2), do.stride(3),
            delta.stride(0), delta.stride(1), delta.stride(2),
            B, H, N_Q,
            BLOCK_M=block_m,
            HEAD_DIM=D,
            num_warps=4,
        )

        # ---- 2. dK, dV pass (outer loop over KV blocks, no atomics) ----
        _bwd_dkdv_kernel[(n_kb, B * H)](
            q, k, v, do, dk, dv, lse, delta,
            block_mask,
            softmax_scale,
            int(sink_size), int(window_size),
            q.stride(0), q.stride(1), q.stride(2), q.stride(3),
            k.stride(0), k.stride(1), k.stride(2), k.stride(3),
            v.stride(0), v.stride(1), v.stride(2), v.stride(3),
            do.stride(0), do.stride(1), do.stride(2), do.stride(3),
            dk.stride(0), dk.stride(1), dk.stride(2), dk.stride(3),
            dv.stride(0), dv.stride(1), dv.stride(2), dv.stride(3),
            lse.stride(0), lse.stride(1), lse.stride(2),
            delta.stride(0), delta.stride(1), delta.stride(2),
            block_mask.stride(0), block_mask.stride(1),
            B, H, N_Q, N_KV,
            BLOCK_M=block_m,
            BLOCK_N=block_n,
            HEAD_DIM=D,
            NUM_Q_BLOCKS=n_qb,
            IS_CAUSAL=is_causal,
            num_warps=4,
            num_stages=2,
        )

        # ---- 3. dQ pass (outer loop over Q blocks) ----
        _bwd_dq_kernel[(n_qb, B * H)](
            q, k, v, do, dq, lse, delta,
            block_mask,
            softmax_scale,
            int(sink_size), int(window_size),
            q.stride(0), q.stride(1), q.stride(2), q.stride(3),
            k.stride(0), k.stride(1), k.stride(2), k.stride(3),
            v.stride(0), v.stride(1), v.stride(2), v.stride(3),
            do.stride(0), do.stride(1), do.stride(2), do.stride(3),
            dq.stride(0), dq.stride(1), dq.stride(2), dq.stride(3),
            lse.stride(0), lse.stride(1), lse.stride(2),
            delta.stride(0), delta.stride(1), delta.stride(2),
            block_mask.stride(0), block_mask.stride(1),
            B, H, N_Q, N_KV,
            BLOCK_M=block_m,
            BLOCK_N=block_n,
            HEAD_DIM=D,
            NUM_KV_BLOCKS=n_kb,
            IS_CAUSAL=is_causal,
            num_warps=4,
            num_stages=2,
        )

        return dq, dk, dv, None, None, None, None, None, None, None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def block_sparse_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    block_mask: torch.Tensor,
    softmax_scale: Optional[float] = None,
    block_m: int = 64,
    block_n: int = 64,
    is_causal: bool = False,
    sink_size: int = 0,
    window_size: int = 0,
) -> torch.Tensor:
    """Differentiable block-sparse Flash-Attention.

    Parameters
    ----------
    q, k, v:
        4-D tensors of shape ``[batch, heads, seq, head_dim]``. Must live
        on CUDA and share dtype. ``head_dim`` must be a power of two in
        ``{16, 32, 64, 128}`` for best codegen.
    block_mask:
        Shape ``[n_q_blocks, n_kv_blocks]``, dtype coercible to int32.
        Visited iff the entry is non-zero; a visited block *still*
        receives the in-block token-level mask when ``is_causal=True``.
    softmax_scale:
        Logit scale. Defaults to :math:`1 / \\sqrt{D}`.
    block_m, block_n:
        Row/column tile sizes. Must divide the sequence lengths.
    is_causal:
        If ``True``, enforces the StreamingLLM token-level mask
        ``(k_pos < sink_size) | ((k_pos <= q_pos) & (k_pos >=
        q_pos - window_size + 1))`` inside every visited block. This
        is required for causal / sliding-window / StreamingLLM
        semantics; without it, diagonal blocks let tokens see the
        future (fix H-1 from docs/architecture_notes.md).
    sink_size, window_size:
        Token-level StreamingLLM parameters. Only consulted when
        ``is_causal=True``.
    """
    if softmax_scale is None:
        softmax_scale = 1.0 / math.sqrt(q.shape[-1])
    return _BlockSparseFlashAttention.apply(
        q.contiguous(), k.contiguous(), v.contiguous(),
        block_mask, float(softmax_scale), int(block_m), int(block_n),
        bool(is_causal), int(sink_size), int(window_size),
    )


# ---------------------------------------------------------------------------
# Reference (eager) implementation — ground-truth for correctness tests.
# ---------------------------------------------------------------------------


def reference_block_sparse_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    block_mask: torch.Tensor,
    softmax_scale: Optional[float] = None,
    block_m: int = 64,
    block_n: int = 64,
    is_causal: bool = False,
    sink_size: int = 0,
    window_size: int = 0,
) -> torch.Tensor:
    """Pure-PyTorch reference for :func:`block_sparse_attention`.

    Materialises the full :math:`[N_Q, N_{KV}]` logit matrix with
    :math:`-\\infty` outside the mask, then calls softmax.
    :math:`\\Theta(N^2)` memory; only exists for correctness checks
    against the Triton kernel on small inputs.

    The effective mask is ``block_mask_lifted_to_tokens AND token_mask``,
    where ``token_mask`` is the token-level StreamingLLM mask when
    ``is_causal=True`` (fix H-2 from docs/architecture_notes.md: the
    oracle must apply the same in-block token mask as the kernel, or
    gradcheck validates like-with-like against the wrong semantics).
    """
    if softmax_scale is None:
        softmax_scale = 1.0 / math.sqrt(q.shape[-1])

    B, H, N_Q, D = q.shape
    _, _, N_KV, _ = k.shape

    # Lift block-level mask to token level.
    lifted = block_mask.to(torch.bool).repeat_interleave(block_m, dim=0)
    lifted = lifted.repeat_interleave(block_n, dim=1)
    lifted = lifted[:N_Q, :N_KV]

    if is_causal:
        q_pos = torch.arange(N_Q, device=q.device)
        k_pos = torch.arange(N_KV, device=q.device)
        sink = k_pos[None, :] < sink_size
        causal = k_pos[None, :] <= q_pos[:, None]
        windowed = k_pos[None, :] >= (q_pos[:, None] - window_size + 1)
        in_block = sink | (causal & windowed)
        effective = lifted & in_block
    else:
        effective = lifted

    scores = torch.matmul(q, k.transpose(-2, -1)) * softmax_scale  # [B,H,N_Q,N_KV]
    scores = scores.masked_fill(~effective, float("-inf"))
    probs = torch.softmax(scores, dim=-1)
    probs = torch.nan_to_num(probs, nan=0.0)
    return torch.matmul(probs, v)


def streaming_token_mask(
    n_q: int, n_kv: int, sink_size: int, window_size: int,
    device: torch.device | str = "cpu",
) -> torch.Tensor:
    """Return the exact token-level StreamingLLM mask.

    Shape ``[n_q, n_kv]``, dtype bool. ``True`` iff the (q, k) token
    pair is attended to under the paper's definition:

    .. math::

        \\mathcal{A}(q) \\;=\\;
        \\{k : k < S\\} \\;\\cup\\; \\{k : q - W + 1 \\le k \\le q\\}

    Independent of any block-level mask. Used by the correctness test
    in :file:`tests/test_mask_equivalence.py` to assert that the
    effective mask the kernel applies (block gate ∩ in-block token
    mask) equals this pure definition.
    """
    q_pos = torch.arange(n_q, device=device)
    k_pos = torch.arange(n_kv, device=device)
    sink = k_pos[None, :] < sink_size
    causal = k_pos[None, :] <= q_pos[:, None]
    windowed = k_pos[None, :] >= (q_pos[:, None] - window_size + 1)
    return sink | (causal & windowed)


__all__ = [
    "StreamingMaskSpec",
    "block_sparse_attention",
    "make_dense_mask",
    "make_streaming_mask",
    "reference_block_sparse_attention",
    "streaming_token_mask",
]
