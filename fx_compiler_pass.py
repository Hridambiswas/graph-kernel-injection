"""Graph-level kernel injection pass for scaled-dot-product attention.

This module ships :class:`BlockSparseAttentionInjector`, a
``torch.fx``-based compiler pass that

1. symbolically traces a user ``nn.Module``,
2. identifies every subgraph that implements the standard
   *scaled-dot-product attention* motif
   :math:`\\text{softmax}(Q K^\\top / \\sqrt{d})\\, V`,
3. folds that subgraph into a single ``call_module`` invocation of
   :class:`_InjectedTritonAttention`, a thin ``nn.Module`` whose
   ``forward`` calls the fused Triton kernel in
   :mod:`sparse_triton_kernel`.

The pass is **topology-preserving**: the IO signature of the traced
module is unchanged, so it can be dropped into an existing training or
inference loop with no downstream edits.

The design intentionally mirrors the way lowering passes are written in
production ML compilers (TorchInductor, OpenXLA, TVM Relay):

* a *pattern-detector* that walks the IR once in :math:`O(|V| + |E|)`
  time, matching against a small fixed templates,
* a *rewriter* that mutates the ``torch.fx.Graph`` under a
  ``graph.inserting_before`` context and preserves def-use consistency,
* a final ``graph.lint`` + ``recompile`` step that validates the
  resulting IR before handing it back to the user.
"""

from __future__ import annotations

import math
import operator
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Tuple

import torch
import torch.fx as fx
import torch.nn as nn

from sparse_triton_kernel import (
    StreamingMaskSpec,
    block_sparse_attention,
    make_streaming_mask,
)


# ---------------------------------------------------------------------------
# Reference eager module — the "input" the FX pass is designed to rewrite.
# ---------------------------------------------------------------------------


class StandardAttention(nn.Module):
    """Textbook scaled-dot-product attention written in eager PyTorch.

    This module exists purely so a test harness can symbolic-trace a
    known-good graph and watch the pass rewrite it. The arithmetic is
    deliberately kept close to the ``torch.nn.functional`` reference:

    .. math::

        \\mathrm{Attn}(Q, K, V) =
        \\text{softmax}\\!\\left(\\frac{Q K^\\top}{\\sqrt{d}}\\right) V

    The materialised :math:`QK^\\top` matrix is :math:`\\Theta(N^2)`
    bytes — the exact "memory wall" we rewrite away.
    """

    def __init__(self, head_dim: int) -> None:
        super().__init__()
        if head_dim <= 0:
            raise ValueError("head_dim must be positive")
        self.head_dim = head_dim
        # Pre-computed to keep the FX graph free of python-level math.
        self.inv_sqrt_d = 1.0 / math.sqrt(head_dim)

    def forward(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor
    ) -> torch.Tensor:
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.inv_sqrt_d
        probs = torch.softmax(scores, dim=-1)
        return torch.matmul(probs, v)


class SingleLayerTransformer(nn.Module):
    """Minimal transformer-ish block wrapping :class:`StandardAttention`.

    Added so the FX pass can be exercised on a module whose graph has
    more than one node, i.e. where attention is a *sub-pattern* and not
    the whole graph.
    """

    def __init__(self, head_dim: int, model_dim: int) -> None:
        super().__init__()
        self.q_proj = nn.Linear(model_dim, model_dim, bias=False)
        self.k_proj = nn.Linear(model_dim, model_dim, bias=False)
        self.v_proj = nn.Linear(model_dim, model_dim, bias=False)
        self.out_proj = nn.Linear(model_dim, model_dim, bias=False)
        self.attn = StandardAttention(head_dim)
        self.num_heads = model_dim // head_dim
        self.head_dim = head_dim

    def _split(self, x: torch.Tensor) -> torch.Tensor:
        B, N, _ = x.shape
        return x.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)

    def _merge(self, x: torch.Tensor) -> torch.Tensor:
        return x.transpose(1, 2).contiguous().flatten(-2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        q = self._split(self.q_proj(x))
        k = self._split(self.k_proj(x))
        v = self._split(self.v_proj(x))
        o = self.attn(q, k, v)
        return self.out_proj(self._merge(o))


# ---------------------------------------------------------------------------
# Injected kernel wrapper — the "output" the FX pass installs.
# ---------------------------------------------------------------------------


class _InjectedTritonAttention(nn.Module):
    """``nn.Module`` wrapper around the Triton block-sparse kernel.

    The FX pass installs this as a child module of the traced
    GraphModule and rewrites the attention subgraph into a single
    ``call_module`` node that targets it.
    """

    def __init__(
        self,
        block_mask: torch.Tensor,
        softmax_scale: float,
        block_m: int,
        block_n: int,
    ) -> None:
        super().__init__()
        self.register_buffer("block_mask", block_mask, persistent=False)
        self.softmax_scale = float(softmax_scale)
        self.block_m = int(block_m)
        self.block_n = int(block_n)

    def forward(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor
    ) -> torch.Tensor:
        return block_sparse_attention(
            q, k, v,
            self.block_mask,
            softmax_scale=self.softmax_scale,
            block_m=self.block_m,
            block_n=self.block_n,
        )


# ---------------------------------------------------------------------------
# The compiler pass itself.
# ---------------------------------------------------------------------------


@dataclass
class InjectionConfig:
    """User-facing configuration for :class:`BlockSparseAttentionInjector`.

    Attributes
    ----------
    seq_len_q, seq_len_kv:
        The query / key sequence lengths the injected kernel is
        specialised for. The block mask is pre-built against these.
    head_dim:
        Head dimension. Used solely to default-compute
        ``softmax_scale = 1 / sqrt(head_dim)``.
    streaming:
        StreamingLLM parameters (sink + window). Supply ``None`` for a
        user-provided ``custom_mask`` instead.
    custom_mask:
        Optional pre-built block mask of shape
        ``[seq_len_q // block_m, seq_len_kv // block_n]``. Takes
        precedence over ``streaming``.
    block_m, block_n:
        Row/column tile sizes passed through to the Triton kernel.
    device:
        Device on which the block-mask buffer is allocated.
    """

    seq_len_q: int
    seq_len_kv: int
    head_dim: int
    streaming: Optional[StreamingMaskSpec] = None
    custom_mask: Optional[torch.Tensor] = None
    block_m: int = 64
    block_n: int = 64
    device: str | torch.device = "cuda"


@dataclass
class _AttentionMatch:
    """A located scaled-dot-product-attention subgraph."""

    q_node: fx.Node
    k_node: fx.Node
    v_node: fx.Node
    # The transpose node (K.transpose) is detached and erased separately.
    transpose_node: fx.Node
    qk_node: fx.Node
    scale_node: Optional[fx.Node]
    softmax_node: fx.Node
    pv_node: fx.Node  # Final matmul node — the match's output.
    scale_value: float


# A node is a "matmul call" if it is either ``torch.matmul`` or the
# ``@`` operator (which fx lowers to :func:`operator.matmul`).
_MATMUL_TARGETS: Tuple[Callable, ...] = (torch.matmul, operator.matmul)

# ``operator.mul`` matches ``x * y``; ``torch.mul`` matches the
# functional form. Division by sqrt(d) also appears as ``operator.truediv``.
_SCALE_TARGETS: Tuple[Callable, ...] = (
    operator.mul, torch.mul, operator.truediv, torch.div,
)


def _is_matmul(node: fx.Node) -> bool:
    return node.op == "call_function" and node.target in _MATMUL_TARGETS


def _is_scale(node: fx.Node) -> bool:
    return node.op == "call_function" and node.target in _SCALE_TARGETS


def _is_softmax(node: fx.Node) -> bool:
    if node.op == "call_function" and node.target in (
        torch.softmax, torch.nn.functional.softmax,
    ):
        return True
    if node.op == "call_method" and node.target == "softmax":
        return True
    return False


def _is_transpose(node: fx.Node) -> bool:
    return (
        node.op == "call_method"
        and node.target in ("transpose", "mT")
    ) or (
        node.op == "call_function" and node.target is torch.transpose
    )


def _extract_scalar(node: fx.Node) -> Optional[float]:
    """Try to pull out the scalar factor from a scale node.

    The scale appears either as a get_attr (e.g. ``self.inv_sqrt_d``)
    or as a bare python float embedded in the fx args. Returns ``None``
    if the scalar cannot be statically determined.
    """
    for arg in node.args:
        if isinstance(arg, (int, float)):
            return float(arg)
        if isinstance(arg, fx.Node) and arg.op == "get_attr":
            return None  # Resolved later by the pass against the owner module.
    return None


class BlockSparseAttentionInjector:
    """Compiler pass that fuses eager attention into the Triton kernel.

    Example
    -------
    >>> model = SingleLayerTransformer(head_dim=64, model_dim=256)
    >>> cfg = InjectionConfig(
    ...     seq_len_q=1024, seq_len_kv=1024, head_dim=64,
    ...     streaming=StreamingMaskSpec(sink_size=4, window_size=256),
    ... )
    >>> injector = BlockSparseAttentionInjector(cfg)
    >>> fused = injector.transform(model)
    """

    def __init__(self, config: InjectionConfig) -> None:
        self.cfg = config
        self._mask = self._build_mask()

    # -- Mask materialisation ------------------------------------------------

    def _build_mask(self) -> torch.Tensor:
        if self.cfg.custom_mask is not None:
            return self.cfg.custom_mask.to(
                device=self.cfg.device, dtype=torch.int32
            ).contiguous()
        if self.cfg.streaming is None:
            raise ValueError("Must supply either `streaming` or `custom_mask`")
        return make_streaming_mask(
            self.cfg.seq_len_q, self.cfg.seq_len_kv,
            self.cfg.streaming,
            self.cfg.block_m, self.cfg.block_n,
            device=self.cfg.device,
        )

    # -- Pattern matching ----------------------------------------------------

    @staticmethod
    def _match_attention_subgraph(
        graph: fx.Graph, owner: fx.GraphModule
    ) -> List[_AttentionMatch]:
        """Enumerate attention subgraphs in a single :math:`O(|V|)` walk.

        The template we look for is::

            T   = K.transpose(-2, -1)
            S   = Q @ T           # matmul
            S'  = S * scale       # or S / sqrt(d)
            P   = softmax(S', -1)
            O   = P @ V           # matmul

        This is deliberately conservative — it refuses to match if any
        node has multiple users whose uses leave the subgraph, which
        would make the rewrite unsafe without duplicating ops.
        """
        matches: List[_AttentionMatch] = []
        for node in graph.nodes:
            if not _is_softmax(node):
                continue
            softmax = node

            # softmax dim must be -1 (standard attention)
            dim = softmax.kwargs.get("dim", None)
            if dim is None and len(softmax.args) >= 2:
                dim = softmax.args[1]
            if dim is not None and dim != -1:
                continue

            scaled_input = softmax.args[0] if softmax.args else None
            if not isinstance(scaled_input, fx.Node):
                continue

            # The input to softmax should be either `S * scale` or raw `S`.
            if _is_scale(scaled_input):
                scale_node: Optional[fx.Node] = scaled_input
                qk_candidate = scaled_input.args[0]
                scale_value = _extract_scalar(scaled_input)
                if scale_value is None:
                    scale_value = BlockSparseAttentionInjector._resolve_scalar(
                        scaled_input, owner
                    )
                if scale_value is None:
                    continue
                if scaled_input.target in (operator.truediv, torch.div):
                    scale_value = 1.0 / scale_value
            else:
                scale_node = None
                qk_candidate = scaled_input
                scale_value = 1.0

            if not (isinstance(qk_candidate, fx.Node) and _is_matmul(qk_candidate)):
                continue
            qk_node = qk_candidate

            q_node, kt_node = qk_node.args[0], qk_node.args[1]
            if not (isinstance(q_node, fx.Node) and isinstance(kt_node, fx.Node)):
                continue
            if not _is_transpose(kt_node):
                continue
            k_node = kt_node.args[0]
            if not isinstance(k_node, fx.Node):
                continue

            # Downstream: softmax must feed into a matmul with V on the RHS.
            users = list(softmax.users)
            if len(users) != 1:
                continue
            pv_node = users[0]
            if not _is_matmul(pv_node):
                continue
            if pv_node.args[0] is not softmax:
                continue
            v_node = pv_node.args[1]
            if not isinstance(v_node, fx.Node):
                continue

            matches.append(_AttentionMatch(
                q_node=q_node, k_node=k_node, v_node=v_node,
                transpose_node=kt_node, qk_node=qk_node,
                scale_node=scale_node, softmax_node=softmax,
                pv_node=pv_node, scale_value=scale_value,
            ))
        return matches

    @staticmethod
    def _resolve_scalar(
        scale_node: fx.Node, owner: fx.GraphModule
    ) -> Optional[float]:
        """Resolve a scale node whose factor is a module attribute."""
        for arg in scale_node.args:
            if isinstance(arg, fx.Node) and arg.op == "get_attr":
                value = getattr(owner, arg.target, None)
                if isinstance(value, (int, float)):
                    return float(value)
                if isinstance(value, torch.Tensor) and value.numel() == 1:
                    return float(value.item())
        return None

    # -- Rewrite -------------------------------------------------------------

    def transform(self, model: nn.Module) -> fx.GraphModule:
        """Trace ``model`` and inject the Triton kernel.

        Returns a new ``fx.GraphModule``; the input ``model`` is left
        untouched.
        """
        traced = fx.symbolic_trace(model)
        matches = self._match_attention_subgraph(traced.graph, traced)
        if not matches:
            raise RuntimeError(
                "No scaled-dot-product-attention subgraph found. "
                "Did you pass a module that wraps StandardAttention?"
            )
        for i, match in enumerate(matches):
            self._install_match(traced, match, index=i)
        traced.graph.lint()
        traced.recompile()
        return traced

    def _install_match(
        self, gm: fx.GraphModule, match: _AttentionMatch, index: int,
    ) -> None:
        """Replace one matched subgraph with a call_module node."""
        name = f"_triton_attn_{index}"
        gm.add_submodule(name, _InjectedTritonAttention(
            block_mask=self._mask,
            softmax_scale=match.scale_value,
            block_m=self.cfg.block_m,
            block_n=self.cfg.block_n,
        ))

        with gm.graph.inserting_before(match.pv_node):
            new_node = gm.graph.call_module(
                name, args=(match.q_node, match.k_node, match.v_node),
            )
        match.pv_node.replace_all_uses_with(new_node)

        # Erase in reverse topological order so def-use edges don't
        # dangle. The transpose node is only safe to erase if nobody
        # else uses it.
        _safe_erase(gm.graph, match.pv_node)
        _safe_erase(gm.graph, match.softmax_node)
        if match.scale_node is not None:
            _safe_erase(gm.graph, match.scale_node)
        _safe_erase(gm.graph, match.qk_node)
        _safe_erase(gm.graph, match.transpose_node)


def _safe_erase(graph: fx.Graph, node: fx.Node) -> None:
    """Erase ``node`` iff it has no remaining users."""
    if len(list(node.users)) == 0:
        graph.erase_node(node)


# ---------------------------------------------------------------------------
# Pretty-printing helpers (for the README and benchmark output).
# ---------------------------------------------------------------------------


def pretty_print_graph(gm: fx.GraphModule, header: str = "") -> str:
    """Return a human-readable rendering of an FX graph.

    Mirrors the format used by production ML compilers in their logging
    output: a header, the op table, and a tally of op kinds.
    """
    out: List[str] = []
    if header:
        out.append(f"=== {header} ===")
    out.append(gm.graph.python_code(root_module="self").src.strip())
    kinds: dict[str, int] = {}
    for node in gm.graph.nodes:
        kinds[node.op] = kinds.get(node.op, 0) + 1
    tally = ", ".join(f"{k}={v}" for k, v in sorted(kinds.items()))
    out.append(f"# node tally: {tally}")
    return "\n".join(out)


__all__ = [
    "BlockSparseAttentionInjector",
    "InjectionConfig",
    "SingleLayerTransformer",
    "StandardAttention",
    "pretty_print_graph",
]
