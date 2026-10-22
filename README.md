# Automated Graph-Level Kernel Injection for Long-Context LLMs

A self-contained Python repository demonstrating a production-grade
*ML-compiler pass*: a hand-written Triton kernel for **block-sparse
Flash-Attention** (forward + backward) and a **PyTorch FX** compiler
pass that identifies eager attention subgraphs in a traced computation
graph and automatically swaps them for the fused kernel.

The repo is intentionally one logical "pull-request" worth of code:
small enough to audit top-to-bottom in a sitting, large enough to be
non-trivial.

```
sparse_triton_kernel.py   # Triton fwd + bwd kernels, autograd bridge, mask builders
fx_compiler_pass.py       # torch.fx pattern-match → rewrite → lint → recompile
benchmark.py              # strict gradcheck + perf harness (GB/s, VRAM, p95)
```

## 1. The memory wall in long-context decoding

For an autoregressive decode step at context length $N$ with head
dimension $D$ (per batch, per head):

| Quantity                      | Eager            | Dense FlashAttention | **This repo (StreamingLLM mask)** |
| :---------------------------- | :--------------: | :------------------: | :-------------------------------: |
| Peak workspace                | $\Theta(N^2)$    | $\Theta(ND)$         | $\Theta(ND)$                      |
| HBM reads (KV)                | $\Theta(ND)$     | $\Theta(ND)$         | $\Theta((S+W)D)$                  |
| Materialised attention matrix | yes              | **no**               | **no**                            |

Modern GPUs are **not compute bound** at decode time. An H100 reads HBM
at ≈3.3 TB/s and does fp16 matmul at ≈1 PF/s; the arithmetic intensity
of attention is low enough that bandwidth dominates. At $N=10^5$,
$D=64$, dense FlashAttention still has to stream the full
$2 \times 100\text{k} \times 64 \times 2 = 25.6$ MB of K+V (fp16) per
head per decode step. The StreamingLLM mask (sink size $S=4$, window
$W=2048$) reduces that to $2 \times 2052 \times 64 \times 2 \approx
525$ kB — a **≈ 49× bandwidth reduction** which, under Little's law,
translates almost linearly into decode speed.

The eager column is included to remind the reader *why* FlashAttention
exists in the first place: the $N \times N$ scratch at $N=10^5$ costs
40 GB in fp32 — far beyond any single-GPU HBM budget. The point of
this repo is to compose both wins — Flash tiling and block sparsity —
and automate the plumbing with a compiler pass.

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
[`fx_compiler_pass.py`](fx_compiler_pass.py) is a single-pass
$O(|V|)$ walk over the FX graph nodes that pattern-matches a
`softmax` node and climbs back through its inputs: `softmax → mul (or
truediv) → matmul → transpose`, plus one forward hop to `matmul → V`.
The rewriter installs an `_InjectedTritonAttention` child module on
the `GraphModule`, inserts a `call_module` node immediately before the
old output matmul, redirects users, and erases the subgraph in reverse
topological order. Finally `graph.lint()` statically checks def-use
consistency and `recompile()` regenerates Python source.

The pass is *idempotent*: a graph that has already been rewritten
contains no softmax→matmul motif and the pass is a no-op.

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

Running `python benchmark.py verify-grads` executes **three** stacked
checks:

1. `torch.autograd.gradcheck` on tiny shapes (B=1, H=2, N=64, D=16) in
   strict fp32 mode (`allow_tf32=False`,
   `float32_matmul_precision="highest"`). Falls up to fp64 if the host
   supports it (SM80+, Triton ≥ 2.1).
2. `torch.autograd.gradgradcheck` for second-order consistency.
3. Direct element-wise allclose vs eager autograd on larger (fp32)
   shapes — the FlashAttention verification protocol.

Any failure raises; the CLI then refuses to run the perf benchmark.
Performance numbers are never reported for an unverified kernel.

## 6. Benchmarks

Placeholder table — populate with your own hardware readings:

| Config (B,H,N<sub>Q</sub>,N<sub>KV</sub>,D,sink,win) | Path   | p50 ms | p95 ms | peak VRAM | HBM GB/s |
| :--------------------------------------------------- | :----- | -----: | -----: | --------: | -------: |
| (1, 32, 1, 131072, 128, 4, 2048)                     | eager  |      — |      — | **OOM**   |        — |
| (1, 32, 1, 131072, 128, 4, 2048)                     | triton |        |        |           |          |
| (1, 32, 1, 32768, 128, 4, 2048)                      | eager  |        |        |           |          |
| (1, 32, 1, 32768, 128, 4, 2048)                      | triton |        |        |           |          |

Benchmark outputs a CSV (`bench_results.csv`) and a Chrome-trace per
path under `traces/` for `chrome://tracing` consumption.

## 7. Install & usage

The Triton path requires a Linux + CUDA host (Triton does not ship for
macOS). Local lint-only sanity passes on any OS.

```bash
cd ~/graph-kernel-injection
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# correctness (gradcheck + fwd allclose) — must pass
python benchmark.py verify-grads
python benchmark.py verify-fwd

# full perf sweep
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
