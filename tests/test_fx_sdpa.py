"""FX pass must match `F.scaled_dot_product_attention(is_causal=True)`.

Prompt 003 step 3. The old matcher pivoted on ``softmax`` and so could
not see SDPA calls at all (SDPA is a single opaque ``call_function``
node). The widened matcher looks directly at SDPA call targets.

These tests are CPU-only: they verify the *rewrite happened in the
graph*, not that the resulting kernel runs. Running the kernel
requires Triton.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa: N812

from fx_compiler_pass import BlockSparseAttentionInjector, InjectionConfig
from sparse_triton_kernel import StreamingMaskSpec


class _CausalSDPAModule(nn.Module):
    def forward(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
    ) -> torch.Tensor:
        return F.scaled_dot_product_attention(q, k, v, is_causal=True)


class _NonCausalSDPAModule(nn.Module):
    def forward(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
    ) -> torch.Tensor:
        return F.scaled_dot_product_attention(q, k, v)


class _MultipleSDPACallsModule(nn.Module):
    def forward(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
    ) -> torch.Tensor:
        a = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        b = F.scaled_dot_product_attention(a, k, v, is_causal=True)
        return b


def _default_cfg() -> InjectionConfig:
    return InjectionConfig(
        seq_len_q=64, seq_len_kv=64, head_dim=16,
        streaming=StreamingMaskSpec(sink_size=4, window_size=16),
        block_m=16, block_n=16,
        device="cpu",
    )


def _count_sdpa_calls(gm) -> int:
    return sum(
        1 for n in gm.graph.nodes
        if n.op == "call_function" and n.target is F.scaled_dot_product_attention
    )


def _count_injected_calls(gm) -> int:
    return sum(
        1 for n in gm.graph.nodes
        if n.op == "call_module" and isinstance(n.target, str)
        and n.target.startswith("_triton_attn_")
    )


def test_fx_rewrites_single_causal_sdpa_call() -> None:
    injector = BlockSparseAttentionInjector(_default_cfg())
    fused = injector.transform(_CausalSDPAModule())
    assert _count_sdpa_calls(fused) == 0, (
        "Causal SDPA call should have been rewritten out of the graph"
    )
    assert _count_injected_calls(fused) == 1, (
        "Exactly one _InjectedTritonAttention call_module should remain"
    )


def test_fx_rewrites_multiple_causal_sdpa_calls() -> None:
    injector = BlockSparseAttentionInjector(_default_cfg())
    fused = injector.transform(_MultipleSDPACallsModule())
    assert _count_sdpa_calls(fused) == 0
    assert _count_injected_calls(fused) == 2


def test_fx_refuses_non_causal_sdpa() -> None:
    injector = BlockSparseAttentionInjector(_default_cfg())
    with pytest.raises(RuntimeError, match="No scaled-dot-product-attention"):
        injector.transform(_NonCausalSDPAModule())


def test_fx_injected_module_carries_streamingllm_params() -> None:
    """Verify the installed child module has the right sink/window."""
    injector = BlockSparseAttentionInjector(_default_cfg())
    fused = injector.transform(_CausalSDPAModule())
    injected = fused.get_submodule("_triton_attn_0")
    assert injected.is_causal is True
    assert injected.sink_size == 4
    assert injected.window_size == 16
