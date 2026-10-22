# Automated Graph-Level Kernel Injection for Long-Context LLMs

## Status

**Not yet run on GPU.** All numeric claims in this README that refer
to specific latencies, memory numbers, bandwidth or speedups are
**analytical** — derived from the mask formula and vendor spec
sheets — and are marked "not yet measured" below. The first hardware
run is scheduled for 2026-10-23 via `scripts/gpu_run.sh`; after that,
this section will be rewritten with the real numbers from
`results/<timestamp>/bench_results.csv`.

Written with heavy AI assistance (Claude Code). **Hridam reviewed:**
TODO. **Hridam verified:** TODO. The repo is a proof-of-work asset
for outreach to ML systems groups; every claim must be either
measured on real hardware or explicitly marked unverified. See
`CLAUDE.md` and `docs/architecture_notes.md` for the honesty rules
and the current correctness audit; see `docs/eval_plan.md` for the
exact shapes and win/tie/loss criteria we will measure.

---

A self-contained Python repository containing a hand-written Triton
kernel for **block-sparse Flash-Attention** (forward + backward) and
a **PyTorch FX** compiler pass that identifies eager attention
subgraphs in a traced computation graph and swaps them for the fused
kernel.

```
sparse_triton_kernel.py   # Triton fwd + bwd kernels, autograd bridge, mask builders
fx_compiler_pass.py       # torch.fx pattern-match → rewrite → lint → recompile
benchmark.py              # strict gradcheck + perf harness (GB/s, VRAM, p95)
tests/                    # CPU-runnable unit tests (mask equivalence, oracle, FX pass)
scripts/gpu_run.sh        # one-command wrapper for the GPU host
```

## 1. The memory wall in long-context decoding

*All figures in this section are **analytical** — see the Status
block above. No per-method latency on this hardware has been
measured yet.*

For an autoregressive decode step at context length $N$ with head
dimension $D$ (per batch, per head):

| Quantity                      | Eager            | Dense FlashAttention | **This repo (StreamingLLM mask)** |
| :---------------------------- | :--------------: | :------------------: | :-------------------------------: |
| Peak workspace                | $\Theta(N^2)$    | $\Theta(ND)$         | $\Theta(ND)$                      |
| HBM reads (KV)                | $\Theta(ND)$     | $\Theta(ND)$         | $\Theta((S+W)D)$                  |
| Materialised attention matrix | yes              | **no**               | **no**                            |

Modern GPUs are typically bandwidth-bound at decode time (the
arithmetic intensity of attention is low); the published vendor
numbers for an H100 are ≈3.3 TB/s of HBM and ≈1 PF/s of fp16
matmul. At $N=10^5$, $D=64$, dense FlashAttention still has to
stream the full $2 \times 100\text{k} \times 64 \times 2 = 25.6$
MB of K+V (fp16) per head per decode step. The StreamingLLM mask
(sink size $S=4$, window $W=2048$) reduces that to $2 \times 2052
\times 64 \times 2 \approx 525$ kB — a **≈ 49× analytical reduction
in KV bytes read**. The actual wall-clock multiplier depends on
whether the comparison is memory-bound in practice at this shape,
and is **not yet measured** — the eval plan at `docs/eval_plan.md`
fixes exactly how we will measure it.

The eager column is included to remind the reader *why*
FlashAttention exists at all: the $N \times N$ scratch at $N=10^5$
is $\approx 40$ GB in fp32 (analytical), far beyond any single-GPU
HBM budget. The repo composes two existing wins (Flash tiling and
block sparsity) and adds a small FX pass that swaps the kernel in
automatically.

## 2. StreamingLLM sparsity pattern

Following Xiao et al. (ICLR 2024), each query token at position $q$
attends to:

$$
\mathcal{A}(q) \;=\;
\underbrace{\{0, 1, \ldots, S-1\}}_{\text{attention sinks}}
\;\cup\;
\underbrace{\{q - W + 1, \ldots, q\}}_{\text{sliding window}}
$$

Lifted to block level with query tiles of size $B_M$ and key tiles of
size $B_N$, block pair $(i, j)$ is kept iff the induced token
rectangles have non-empty intersection with $\mathcal{A}$. The mask is
built once by
[`make_streaming_mask`](sparse_triton_kernel.py) in
$O(n_q n_{kv} / (B_M B_N))$ work and consumes
$O(n_q n_{kv} / (B_M B_N))$ int32 bytes — negligible against the KV
cache itself.

## 3. The FX compiler pass

`BlockSparseAttentionInjector` is a one-shot pass. The input is any
`nn.Module` whose forward contains the canonical scaled-dot-product
attention motif; the output is a `torch.fx.GraphModule` with that
motif replaced by a single `call_module` node.

### Before — eager attention

```
%scores     : call_function[target=torch.matmul](%q, %kT)
%scaled     : call_function[target=operator.mul](%scores, %inv_sqrt_d)
%probs      : call_function[target=torch.softmax](%scaled, dim=-1)
%out        : call_function[target=torch.matmul](%probs, %v)
```

### After — fused kernel call

```
%out        : call_module[target=_triton_attn_0](%q, %k, %v)
```

The pattern detector in
[`fx_compiler_pass.py`](fx_compiler_pass.py) does two passes over
the FX graph nodes:

1. A single-pass $O(|V|)$ walk pivoting on `softmax` nodes,
   climbing back through `softmax → mul (or truediv) → matmul →
   transpose`, plus one forward hop to `matmul → V`. Matches the
   repo's own hand-written `StandardAttention` motif.
2. A second $O(|V|)$ walk for `F.scaled_dot_product_attention`
   call_function nodes with `is_causal=True` and `dropout_p=0`.
   Added by prompt 003 so models that already use SDPA are also
   matched; non-causal SDPA is deliberately refused for now because
   the kernel can't absorb a dynamic `attn_mask` tensor.

The rewriter installs an `_InjectedTritonAttention` child module on
the `GraphModule`, inserts a `call_module` node immediately before
the matched output node, redirects users, and erases the subgraph.
Matches are installed in reverse topological order so chained
attention calls don't leave stale references in the dataclass.
Finally `graph.lint()` statically checks def-use consistency and
`recompile()` regenerates Python source.

Not matched: `torch.nn.MultiheadAttention`, Hugging Face transformer
attention classes. No test exists for either; see
`docs/architecture_notes.md §2` for the specifics.

## 4. The Triton kernel

### SRAM tile layout

```
                         ┌──────────── BLOCK_N ───────────┐
                         │                                │
     ┌──── D ────┐       ┌─ K tile (streaming) ───────────┐
┌─┐  │           │       │                                │
│Q│  │  BLOCK_M  │       │                                │
│ │  │           │       └────────────────────────────────┘
└─┘  └───────────┘       ┌─ V tile (streaming) ───────────┐
     register-resident   │                                │
     for the whole       └────────────────────────────────┘
     kv reduction        ↑
                         issued only if block_mask[i,j] == 1
```

The forward kernel keeps `Q`, the running softmax statistics
`(m_i, l_i)`, and the accumulator `acc` register-resident across the
inner reduction. Each program instance owns one query tile and
iterates over the key/value tiles **gated by the block mask**: a single
int32 load decides whether to issue the much more expensive K+V block
loads. Online softmax (FlashAttention-2 numerics) keeps the state
numerically stable across the reduction.

### Why two backward kernels

A naive single-pass backward would race on $\partial Q$ because
multiple $(i, j)$ pairs contribute to the same query row. Rather than
pay for `atomicAdd`s, we follow FlashAttention v2 and split the
backward into two mirror-image kernels:

* `_bwd_dkdv_kernel` — outer loop over *KV* blocks. $\partial K_j$ and
  $\partial V_j$ receive contributions from every $(i, j)$ with
  `block_mask[i, j] = 1`, which this kernel owns exclusively → no
  atomics.
* `_bwd_dq_kernel` — outer loop over *Q* blocks. Mirror image. $\partial
  Q_i$ receives contributions from every $j$ with `block_mask[i, j] = 1`.

A tiny `_bwd_preprocess_kernel` materialises $D_i = \langle dO_i, O_i
\rangle$ once so the two main backward kernels don't redundantly
recompute it.

## 5. Correctness

Running `python benchmark.py verify-grads` executes three stacked
checks and gates the perf benchmark behind them. The CLI tries the
tightest tier it can and prints which path it took:

1. `torch.autograd.gradcheck` + `gradgradcheck` on tiny shapes
   (B=1, H=2, N=64, D=16). First tier: **fp64** with
   `eps=1e-6, atol=1e-5, rtol=1e-3`. This path requires a GPU
   whose `tl.dot` supports fp64 — typically only datacenter cards
   (A100, H100). On consumer / laptop GPUs the fp64 path raises at
   the Triton level and the harness falls back to tier 2 with a
   clearly logged notice.
2. **fp32 FA2 protocol**: same shapes in strict fp32
   (`allow_tf32=False`, `float32_matmul_precision="highest"`) with
   `eps=1e-3, atol=5e-3, rtol=5e-3`.
3. Direct element-wise allclose of `(dQ, dK, dV)` vs eager autograd
   on larger (fp32) shapes.

Any failure raises; the CLI exits non-zero and refuses to run the
perf benchmark. The specific tier taken is printed in the log so a
reader can tell whether fp64 was actually exercised or whether the
fallback kicked in. In addition, 28 CPU-runnable unit tests under
`tests/` cover the token-level mask equivalence (H-1 / H-2 fix) and
the FX pass's SDPA matching — these do not require Triton or a GPU.

## 6. Benchmarks

**No numbers are published here yet.** After the first GPU run under
`scripts/gpu_run.sh`, this section will be rewritten verbatim from
`results/<timestamp>/bench_results.csv`. The exact shapes we plan to
measure, the win/tie/loss criteria vs FlexAttention, and the
honesty disclaimer about what we are *not* claiming are specified
in [`docs/eval_plan.md`](docs/eval_plan.md).

The benchmark sweep compares five methods on the **same**
token-level StreamingLLM mask: `triton-block-sparse` (this repo),
`flex-attention` (torch 2.5+), `sdpa-explicit-mask` (SDPA with the
same boolean mask), `sdpa-causal-dense` (plain causal SDPA — "no
sparsity" reference, not StreamingLLM), and `eager-dense`
(reference oracle). Per-method output: fwd p50/p95, fwd+bwd
p50/p95, peak VRAM, and max abs error vs an fp64 oracle at a
scaled-down shape.

Benchmark outputs a CSV (`bench_results.csv`) and a Chrome-trace
per method under `traces/`.

## 7. Install & usage

The Triton path requires a Linux + CUDA host (Triton does not ship
for macOS). Local lint-only sanity + the CPU-runnable unit tests
pass on any OS.

```bash
cd ~/graph-kernel-injection
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# CPU-only tests (no Triton required) — must pass
python -m pytest tests/

# everything else requires an NVIDIA GPU on Linux:
bash scripts/gpu_run.sh          # full-size shapes
bash scripts/gpu_run.sh --small  # trimmed for an 8 GiB laptop GPU

# or call benchmark.py directly:
python benchmark.py verify-grads
python benchmark.py verify-fwd
python benchmark.py benchmark --seq-len 131072 --sink 4 --window 2048 --dtype fp16
```

Using the FX pass from Python:

```python
from fx_compiler_pass import (
    BlockSparseAttentionInjector, InjectionConfig,
    SingleLayerTransformer, pretty_print_graph,
)
from sparse_triton_kernel import StreamingMaskSpec

model = SingleLayerTransformer(head_dim=64, model_dim=512).cuda()
cfg = InjectionConfig(
    seq_len_q=1024, seq_len_kv=1024, head_dim=64,
    streaming=StreamingMaskSpec(sink_size=4, window_size=256),
)
fused = BlockSparseAttentionInjector(cfg).transform(model)
print(pretty_print_graph(fused, header="after injection"))
```

## 8. References

* Dao, T. *FlashAttention-2: Faster Attention with Better Parallelism
  and Work Partitioning.* arXiv:2307.08691, 2023.
* Xiao, G. et al. *Efficient Streaming Language Models with Attention
  Sinks.* ICLR 2024.
* Tillet, P. et al. *Triton: An Intermediate Language and Compiler for
  Tiled Neural Network Computations.* MAPL 2019.
* Reed, J. et al. *torch.fx: Practical Program Capture and
  Transformation for Deep Learning in Python.* MLSys 2022.

## License

MIT.
