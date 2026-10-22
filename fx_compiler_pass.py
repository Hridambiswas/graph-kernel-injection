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
import torch.nn.functional as F  # noqa: N812

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
        is_causal: bool = False,
        sink_size: int = 0,
        window_size: int = 0,
    ) -> None:
        super().__init__()
        self.register_buffer("block_mask", block_mask, persistent=False)
        self.softmax_scale = float(softmax_scale)
        self.block_m = int(block_m)
        self.block_n = int(block_n)
        self.is_causal = bool(is_causal)
        self.sink_size = int(sink_size)
        self.window_size = int(window_size)

    def forward(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor
    ) -> torch.Tensor:
        return block_sparse_attention(
            q, k, v,
            self.block_mask,
            softmax_scale=self.softmax_scale,
            block_m=self.block_m,
            block_n=self.block_n,
            is_causal=self.is_causal,
            sink_size=self.sink_size,
            window_size=self.window_size,
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
    """A located scaled-dot-product-attention subgraph (eager form)."""

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


@dataclass
class _SDPAMatch:
    """A located ``F.scaled_dot_product_attention`` call to replace.

    Prompt 003 step 3: the FX pass used to only match the hand-written
    softmax(matmul(scale(matmul))) motif. Real models route attention
    through ``F.scaled_dot_product_attention``, which is a single
    opaque ``call_function`` node — no softmax to pivot on.
    """

    sdpa_node: fx.Node
    q_node: fx.Node
    k_node: fx.Node
    v_node: fx.Node
    is_causal: bool
    scale_value: Optional[float]  # None → use injector default (1/√D)


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


# F.scaled_dot_product_attention compares by identity under fx.
# Both the public and the internal entry-points resolve to the same
# function object in torch >= 2.0, but we check both for robustness
# across versions / monkeypatches.
_SDPA_TARGETS: Tuple[Callable, ...] = tuple(
    t for t in (
        F.scaled_dot_product_attention,
        getattr(getattr(torch, "_C", None), "_nn", None),
    )
    if callable(t)
)


def _is_sdpa(node: fx.Node) -> bool:
    return node.op == "call_function" and node.target in _SDPA_TARGETS


def _sdpa_param(node: fx.Node, name: str, position: int, default: object) -> object:
    """Extract an SDPA kwarg either from kwargs or from positional args.

    SDPA signature:
        scaled_dot_product_attention(query, key, value,
                                     attn_mask=None, dropout_p=0.0,
                                     is_causal=False, scale=None,
                                     enable_gqa=False)
    """
    if name in node.kwargs:
        return node.kwargs[name]
    if len(node.args) > position:
        return node.args[position]
    return default


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

        Matches both:
          - the hand-written ``softmax(matmul(scale(matmul)))`` motif,
          - ``F.scaled_dot_product_attention(..., is_causal=True)``.

        Returns a new ``fx.GraphModule``; the input ``model`` is left
        untouched.
        """
        traced = fx.symbolic_trace(model)
        eager_matches = self._match_attention_subgraph(traced.graph, traced)
        sdpa_matches = self._match_sdpa_subgraph(traced.graph)
        if not eager_matches and not sdpa_matches:
            raise RuntimeError(
                "No scaled-dot-product-attention subgraph found "
                "(neither the eager softmax(QKᵀ·s)V motif nor an "
                "F.scaled_dot_product_attention(is_causal=True) call)."
            )
        # Capture final indices now (they must be contiguous starting at 0
        # for the submodule naming), then rewrite in reverse topological
        # order so a downstream SDPA call still sees its upstream SDPA's
        # original node until we touch it, letting replace_all_uses_with
        # patch the def-use edge rather than leaving a stale reference.
        eager_indexed = list(enumerate(eager_matches))
        sdpa_start = len(eager_matches)
        sdpa_indexed = [(sdpa_start + i, m) for i, m in enumerate(sdpa_matches)]
        for idx, match in reversed(sdpa_indexed):
            self._install_sdpa_match(traced, match, index=idx)
        for idx, match in reversed(eager_indexed):
            self._install_match(traced, match, index=idx)
        traced.graph.lint()
        traced.recompile()
        return traced

    # -- SDPA matching -------------------------------------------------------

    @staticmethod
    def _match_sdpa_subgraph(graph: fx.Graph) -> List[_SDPAMatch]:
        """Enumerate `F.scaled_dot_product_attention(is_causal=True)` calls."""
        matches: List[_SDPAMatch] = []
        for node in graph.nodes:
            if not _is_sdpa(node):
                continue
            if len(node.args) < 3:
                continue
            q, k, v = node.args[0], node.args[1], node.args[2]
            if not (isinstance(q, fx.Node)
                    and isinstance(k, fx.Node)
                    and isinstance(v, fx.Node)):
                continue
            is_causal = _sdpa_param(node, "is_causal", 5, False)
            if not bool(is_causal):
                # We refuse to match non-causal SDPA: it may rely on
                # an attn_mask that is a dynamic tensor the kernel
                # cannot absorb into its static block mask. A later
                # prompt can add explicit-mask matching.
                continue
            dropout_p = _sdpa_param(node, "dropout_p", 4, 0.0)
            if isinstance(dropout_p, (int, float)) and float(dropout_p) != 0.0:
                continue  # dropout != 0 changes semantics; refuse.
            scale = _sdpa_param(node, "scale", 6, None)
            scale_value: Optional[float]
            if scale is None:
                scale_value = None
            elif isinstance(scale, (int, float)):
                scale_value = float(scale)
            else:
                continue  # dynamic scale we can't resolve; skip.
            matches.append(_SDPAMatch(
                sdpa_node=node, q_node=q, k_node=k, v_node=v,
                is_causal=True, scale_value=scale_value,
            ))
        return matches

    def _install_sdpa_match(
        self, gm: fx.GraphModule, match: _SDPAMatch, index: int,
    ) -> None:
        name = f"_triton_attn_{index}"
        default_scale = 1.0 / math.sqrt(self.cfg.head_dim)
        scale = match.scale_value if match.scale_value is not None else default_scale
        is_causal = match.is_causal
        sink_size = self.cfg.streaming.sink_size if self.cfg.streaming else 0
        # If the user asked for SDPA causal but didn't configure
        # StreamingLLM, interpret as "pure causal" by picking a
        # window large enough to admit every past token.
        window_size = (
            self.cfg.streaming.window_size if self.cfg.streaming
            else self.cfg.seq_len_kv
        )
        gm.add_submodule(name, _InjectedTritonAttention(
            block_mask=self._mask,
            softmax_scale=scale,
            block_m=self.cfg.block_m,
            block_n=self.cfg.block_n,
            is_causal=is_causal,
            sink_size=sink_size,
            window_size=window_size,
        ))
        with gm.graph.inserting_before(match.sdpa_node):
            new_node = gm.graph.call_module(
                name, args=(match.q_node, match.k_node, match.v_node),
            )
        match.sdpa_node.replace_all_uses_with(new_node)
        _safe_erase(gm.graph, match.sdpa_node)

    def _install_match(
        self, gm: fx.GraphModule, match: _AttentionMatch, index: int,
    ) -> None:
        """Replace one matched subgraph with a call_module node."""
        name = f"_triton_attn_{index}"
        is_causal = self.cfg.streaming is not None
        sink_size = self.cfg.streaming.sink_size if is_causal else 0
        window_size = (
            self.cfg.streaming.window_size if is_causal else 0
        )
        gm.add_submodule(name, _InjectedTritonAttention(
            block_mask=self._mask,
            softmax_scale=match.scale_value,
            block_m=self.cfg.block_m,
            block_n=self.cfg.block_n,
            is_causal=is_causal,
            sink_size=sink_size,
            window_size=window_size,
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
