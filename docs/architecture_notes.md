# Architecture notes

Written 2026-10-23 while executing mailbox prompt 001. The repo was written in
one session on 2026-10-22 on a Mac (no NVIDIA GPU on the host). Nothing in it
has been executed on GPU yet. These notes describe what the code *claims* to do
by reading, so later prompts can target real problems rather than guess.

## 1. Modules and how they connect

### `sparse_triton_kernel.py` (~780 lines)

Four logical sections:

**(a) Mask construction utilities.** `StreamingMaskSpec` dataclass holds
`sink_size` and `window_size`. `make_streaming_mask(n_q, n_kv, spec, block_m,
block_n, device)` builds a *block-level* `[n_q // block_m, n_kv // block_n]`
int32 mask: pair `(i, j)` is 1 if any token-level StreamingLLM pair
`(sink_token ∪ causal_window)` falls inside the block rectangle. Requires
`n_q % block_m == 0` and `n_kv % block_n == 0` (raises ValueError otherwise).
`make_dense_mask(…, causal=False)` provides a fully-dense (or dense-causal)
block mask for correctness oracles.

**(b) Triton kernels.** Only compiled if `import triton` succeeds; otherwise
the symbols are omitted and `_BlockSparseFlashAttention.forward` raises at
runtime. Four kernels:

- `_fwd_kernel`. Grid `(num_q_blocks, B*H)`. Loads one `[BLOCK_M, D]` Q tile
  per program, keeps `(m_i, l_i, acc)` register-resident, iterates over
  `NUM_KV_BLOCKS` KV tiles gated by a single int32 load of `BlockMask[pid_m,
  kv_idx]`. If the gate is non-zero, loads K/V, computes `qk = (Q @ K) *
  scale`, applies `tl.where(kv_mask, qk, -inf)` for N_KV bounds, runs online
  FlashAttention-v2 softmax, accumulates `acc = acc * alpha + P @ V`. Stores
  `acc / safe_l` into `Out` and `m_i + log(safe_l)` into `Lse`.
- `_bwd_preprocess_kernel`. Row-wise `D_i = <dO_i, O_i>` into `Delta`.
- `_bwd_dkdv_kernel`. Grid `(num_kv_blocks, B*H)`. Outer loop over Q blocks;
  owns the full reduction for one K/V tile so `dK/dV` can be stored once at
  the end with no atomics. Recomputes `qk`, `p = exp(qk - lse)`, derives
  `ds = p * (dp - delta) * scale`, writes back `dK/dV`.
- `_bwd_dq_kernel`. Mirror image: outer loop over KV blocks, writes `dQ`.

**(c) Autograd bridge.** `_BlockSparseFlashAttention(torch.autograd.Function)`
with `forward` and `backward` static methods. Forward launches `_fwd_kernel`;
saves `(q, k, v, o, lse, block_mask)` and `softmax_scale, block_m, block_n`.
Backward launches the three bwd kernels in sequence
(preprocess → dKdV → dQ). Returns `(dq, dk, dv, None, None, None, None)` to
match the forward signature.

**(d) Public API.** `block_sparse_attention(q, k, v, block_mask,
softmax_scale=None, block_m=64, block_n=64)` — thin wrapper that defaults
`softmax_scale = 1/√D` and calls the autograd Function. `.contiguous()` is
forced on Q, K, V. `reference_block_sparse_attention(...)` materialises the
full `[N_Q, N_KV]` logit matrix with `-inf` outside the token-level mask
derived by `repeat_interleave(block_m)` × `repeat_interleave(block_n)` on the
block mask, then calls `torch.softmax` + matmul V. Serves as the ground-truth
oracle for correctness tests.

### `fx_compiler_pass.py` (~490 lines)

**(a) Example modules.** `StandardAttention(nn.Module)` is the canonical
`softmax(Q @ K.T * scale) @ V` eager pattern. `SingleLayerTransformer`
wraps q/k/v/out `nn.Linear` projections around it. Both exist as *test
targets* for the pass.

**(b) Injection config.** `InjectionConfig` dataclass (seq_len_q, seq_len_kv,
head_dim, streaming or custom_mask, block_m/n, device). The default device is
`"cuda"`; constructing `BlockSparseAttentionInjector(cfg)` immediately
builds the mask on that device, so a CPU-only host must pass `device="cpu"`.

**(c) The pass.** `BlockSparseAttentionInjector` does three things in
`.transform(model)`:
1. `fx.symbolic_trace(model)` to get a `GraphModule`.
2. `_match_attention_subgraph(graph, owner)` — one `O(|V|)` sweep over the
   graph nodes, pivoting on `softmax`. For each softmax node it climbs:
   `softmax` → `mul/truediv` (optional) → `matmul` → `transpose`. The scalar
   factor is pulled either from a literal arg or from a `get_attr` resolved
   against the owner module. Downstream it checks softmax has exactly one
   user and that user is a `matmul` with softmax on the left.
3. `_install_match` — adds an `_InjectedTritonAttention` child module to the
   GraphModule, inserts a `call_module` node before the output matmul, redirects
   users with `replace_all_uses_with`, and erases the subgraph in reverse
   topological order with `_safe_erase` (only erases if the node has no users
   left).
4. `graph.lint()` + `recompile()`.

**(d) `_InjectedTritonAttention`.** `nn.Module` whose `forward(q, k, v)` calls
`block_sparse_attention` with a `block_mask` buffer and the extracted
`softmax_scale`.

### `benchmark.py` (~660 lines)

Three CLI modes selected positionally: `verify-grads`, `verify-fwd`,
`benchmark` (and `all`). `main()` runs them in order and gates perf behind
grad correctness.

- `verify_forward_correctness` — allclose Triton vs reference at
  `256×256×D=64` for fp32 and fp16 on a hard-coded sub-config.
- `verify_gradients` — three stacked checks:
  1. `torch.autograd.gradcheck` (and `gradgradcheck`) at B=1 H=2 N=64 D=16,
     wrapped in a try/except: tries fp64 first (`eps=1e-6, atol=1e-5,
     rtol=1e-3`) and on any exception degrades to fp32 FA2 protocol
     (`eps=1e-3, atol=5e-3, rtol=5e-3`). The fallback is announced in stdout.
  2. Direct `allclose(dQ/dK/dV, eager autograd)` at B=1 H=4 N=512 D=64 fp32
     with `atol=1e-3, rtol=1e-3`.
- `run_benchmark` — makes Q/K/V/mask once, measures Triton and (if it fits)
  dense-eager latency with `torch.cuda.Event`, computes peak VRAM and an
  analytical bytes-moved → GB/s, exports a `chrome://tracing` JSON per path
  under `./traces/`, writes `bench_results.csv`, prints a comparison table
  including VRAM / HBM / p95 ratios.

### Data flow across the three files

```
                 ┌─────── user nn.Module ────────┐
                 │                                │
                 ▼                                │
    fx_compiler_pass.BlockSparseAttentionInjector │
         .transform()                             │
                 │                                │
                 ▼                                │
    GraphModule with _InjectedTritonAttention ─── └── forward(q,k,v)
                 │                                     calls:
                 ▼
         sparse_triton_kernel.block_sparse_attention
                 │
                 ▼
      _BlockSparseFlashAttention.apply(...)   ◄── autograd bridge
                 │
    ┌────────────┼────────────┐
    ▼            ▼            ▼
 _fwd_kernel  _bwd_preprocess  _bwd_dkdv / _bwd_dq
```

`benchmark.py` sits off to the side; it does not use the FX pass. It drives
`block_sparse_attention` directly against `reference_block_sparse_attention`
to exercise the kernel in isolation.

## 2. What the FX pass can and cannot match today

Verified by reading the matcher at `fx_compiler_pass.py:302-386`.

**Matches (confirmed by code reading):**

- The repo's own `StandardAttention` motif, which fx symbolic-traces to:
  ```
  %kT   = call_method[target=transpose](%k, -2, -1)
  %qk   = call_function[target=torch.matmul](%q, %kT)
  %scl  = call_function[target=operator.mul](%qk, %inv_sqrt_d)
  %p    = call_function[target=torch.softmax](%scl, dim=-1)
  %o    = call_function[target=torch.matmul](%p, %v)
  ```
- `SingleLayerTransformer`, where `StandardAttention` is a nested submodule.
  (FX inlines `StandardAttention.forward` into the parent graph, so the four
  attention nodes become siblings of the surrounding `linear` nodes — the
  matcher still finds them.)
- Variants where the softmax is called as a method (`scores.softmax(dim=-1)`)
  or as `torch.nn.functional.softmax(..., dim=-1)` — because `_is_softmax`
  accepts `call_method target='softmax'`, `torch.softmax`, and `F.softmax`.
- Variants where the scale is a Python float literal, a module `get_attr`
  attribute, or a `truediv` (division by `sqrt(d)` resolves to
  `operator.truediv`, which the matcher inverts to `1/x`).
- Variants with **no** scale node at all (unscaled attention): the matcher
  falls through to `scale_node = None, scale_value = 1.0`.

**Does NOT match (also confirmed by code reading):**

- `torch.nn.functional.scaled_dot_product_attention` or
  `torch.nn.attention.scaled_dot_product_attention`. These are a *single*
  `call_function` node; the matcher pivots on `softmax`, which does not
  exist in the traced graph for SDPA. No test exists for this today.
- `torch.nn.MultiheadAttention`. The module cannot be `symbolic_trace`'d
  cleanly because its forward has data-dependent control flow (key_padding
  masks, `need_weights` branches, etc.) and because recent PyTorch versions
  dispatch its inner attention through SDPA, which again is a single opaque
  node. No test exists.
- Hugging Face GPT-2 / Llama attention modules. They are not even in the
  deps, and recent HF models increasingly route through SDPA for the fast
  path; even where they don't, the surrounding dropout, mask addition and
  position-embedding ops break the strict `softmax(matmul(scale(matmul)))`
  pattern. No test exists; the README does not claim support, but a reader
  might assume so.
- Variants where the scale node has operands reversed, i.e.
  `scale * scores` (operator.mul with args[0]=scalar). The matcher reads
  `qk_candidate = scaled_input.args[0]` so it would try to treat the scalar
  as a matmul node and fall through. `scores * scale` is the only form
  matched.
- Softmax with `dim` != -1 (even `dim=3` on a 4-D tensor, which is
  numerically identical). The matcher short-circuits on `dim != -1`.
- A matmul that uses `K.permute(...)` or `K.T` property access instead of
  `K.transpose(-2, -1)` — unless FX lowers `.T`/`.mT` to a `transpose`
  call, which depends on PyTorch's internals.

**Blunt summary:** today the FX pass is a demonstration that the plumbing
works on *the repo's own hand-written attention module*. It does not yet
show that the pass is useful on real models.

## 3. Unmeasured claims in README.md

Line numbers reference the current README (as of the commit audited).

| Line | Claim | Status |
| ---: | :---- | :---- |
| 2    | "production-grade *ML-compiler pass*"                                     | Adjective; nothing measured.               |
| 10-11| "small enough to audit top-to-bottom in a sitting, large enough to be non-trivial" | Subjective framing.                        |
| 24-28| Memory/HBM table (eager/flash/this)                                       | Analytical derivation only, not measured.  |
| 30-32| "H100 reads HBM at ≈3.3 TB/s and does fp16 matmul at ≈1 PF/s"             | Vendor spec, no measurement here.          |
| 36-38| "≈ 49× bandwidth reduction … translates almost linearly into decode speed"| Derived from the mask formula; no benchmark number backs the "almost linearly" claim. |
| 42-44| "40 GB in fp32 — far beyond any single-GPU HBM budget"                    | Arithmetic from N=10⁵, D=64. Not measured. |
| 44   | "automate the plumbing with a compiler pass"                              | Pass exists but is only tested on the repo's own module. |
| 123-128 | "online softmax (FlashAttention-2 numerics) keeps the state numerically stable" | True by construction but no numerical experiment on this machine. |
| 100-101 | "the pass is *idempotent*: a graph that has already been rewritten contains no softmax→matmul motif and the pass is a no-op." | Correct in principle; `.transform` actually raises `RuntimeError("No scaled-dot-product-attention subgraph found. …")` if the pattern is absent, so the second call is NOT a no-op, it crashes. |
| 149-161| "**three** stacked checks … The CLI then refuses to run the perf benchmark" | True in code, but never *actually* run anywhere. |
| 165-172| Placeholder benchmark table                                             | Explicitly labelled "placeholder", but readers may miss that. Rows are empty but the header prints real-looking shapes. |
| 215-222 | References                                                             | Fine. Verified these are real papers.      |

The adjective "production-grade" (line 2) is the single most load-bearing
unmeasured word in the README. On a repo that has never run on hardware it
should be removed; prompt 003 covers that cleanup.

## 4. Correctness audit (reading only; no execution)

Grading: **H**igh = will produce wrong numbers or crash on first GPU
contact; **M**edium = depends on inputs; **L**ow = cosmetic / robustness
only. Everything below is diagnosis — fixes come in prompt 003.

### H-1. Causality is enforced only at block granularity

- Where: `_fwd_kernel` line 285-287, `_bwd_dkdv_kernel` line 430-431,
  `_bwd_dq_kernel` line 523-524. Each does
  `qk = tl.where(bounds_mask, qk, -inf)` with a *bounds* mask (N_Q/N_KV
  only). None applies a causal (`k ≤ q`) mask inside a block.
- Why that matters: StreamingLLM is causal. A query at position `q`
  attends to `{0..S-1} ∪ {q-W+1..q}` — never to tokens at positions
  `> q`. `make_streaming_mask` keeps a block pair `(i, j)` if *any* query
  in block i attends to *any* key in block j; so on diagonal and
  near-diagonal blocks the block mask contains cells `(q, k)` with `k > q`
  that the token-level mask would forbid. The kernel then lets those
  cells into the softmax.
- `reference_block_sparse_attention` lifts the *block* mask to a token
  mask by `repeat_interleave`, so the oracle has the exact same leak. The
  gradcheck therefore compares like with like, and will pass even though
  the kernel does not implement real StreamingLLM semantics.
- Risk: **High**. This is the correctness bug prompt 001 explicitly asks
  about ("tokens can see the future inside a block"). A fix needs (a) a
  token-level causal mask applied inside the kernel for diagonal blocks
  (and more broadly: `k_pos ≤ q_pos`, plus `k_pos ≥ q_pos - W + 1 OR k_pos
  < S`), and (b) the oracle must be updated to use the same token-level
  mask so gradcheck remains honest.

### H-2. Oracle doesn't exercise the sparsity semantics the paper describes

- Where: `reference_block_sparse_attention` (sparse_triton_kernel.py:739-772).
- The oracle "expands the block mask to token level" rather than
  "constructs the StreamingLLM token mask from scratch". The two are
  **not equal** on diagonal blocks (see H-1).
- Risk: **High**. Direct consequence of H-1: the correctness tests
  silently validate an incorrect model of attention. Fix is to build the
  token-level StreamingLLM mask directly in the oracle, independently
  of the block mask, so the oracle encodes the semantics the paper
  defines.

### M-1. Mask constructor requires divisibility, autograd bridge does not

- `make_streaming_mask` lines 141-144 raise if `n_q % block_m != 0` or
  `n_kv % block_n != 0`.
- `_BlockSparseFlashAttention.forward` (line 573-574) uses *ceiling*
  division when computing expected mask shape:
  `n_qb = (N_Q + block_m - 1) // block_m`.
- The kernel's row/col bounds checks are correct, but there is no way to
  actually feed it a non-divisible shape because the mask builder won't
  construct a mask for one. Either the kernel should document the
  divisibility requirement, or the mask builder should allow partial
  edge blocks.
- Risk: **Medium**. Pure inconsistency; won't crash because the mask
  builder shouts first. Fix: pick one convention.

### M-2. Fully-masked rows → `-inf` LSE → NaN in backward

- `_fwd_kernel` guards the output division with
  `safe_l = tl.where(l_i > 0.0, l_i, 1.0)` (line 304), but the LSE store
  is `lse = m_i + tl.log(safe_l)` (line 312). For a fully-masked row,
  `m_i` is still `-inf`, so lse is `-inf`.
- In `_bwd_dkdv_kernel` (and `_bwd_dq_kernel`), `p = tl.exp(qk - lse)`.
  If both `qk` and `lse` are `-inf`, `qk - lse = NaN`; `exp(NaN) = NaN`.
  The NaN flows into dK/dV/dQ.
- A well-formed StreamingLLM mask with `sink_size ≥ 1` keeps block
  column 0 for every row, so this does not fire in practice. But any
  user who passes a custom mask with an empty row (via `custom_mask=` on
  the injector, or a sink_size=0 window_size=0 spec), or any future bug
  that produces an empty row for an edge case, poisons the backward.
- Risk: **Medium**. Fix: either store `0` for lse when `l_i == 0`, or
  skip the row entirely in bwd when `delta == 0 AND lse == -inf`.

### M-3. gradcheck's fp64 path silently falls back to fp32 on most GPUs

- Where: `benchmark.py` lines 287-296. The entire fp64 attempt is
  wrapped in `try/except Exception`, which catches *any* failure — most
  commonly `tl.dot` not supporting fp64 on consumer GPUs (RTX
  40-series, laptop Ada / Ampere, 50-series).
- When the fallback kicks in, it prints a human-readable notice but the
  CLI exit code is 0 either way. A reader of the exit code or the
  "[gradcheck] ✓ fp32 FA2 fallback completed" line could easily miss
  that fp64 was skipped.
- Impact: fp32 gradcheck is still a valid Jacobian check; its tolerances
  (`atol=5e-3, rtol=5e-3`) match the FlashAttention protocol. It is
  weaker, though, and the README (lines 149-158) says "falls up to fp64
  if the host supports it" without qualifying what "support" means.
- Risk: **Medium**. Fix: either (a) record which path was taken in
  the result and surface it in the final summary, or (b) make the fp64
  path a soft precondition (print a WARNING, carry on with fp32) rather
  than silently branch inside a try/except.

### L-1. Scale operand order is fragile

- `_match_attention_subgraph` reads `qk_candidate = scaled_input.args[0]`
  (line 339). If a user writes `self.inv_sqrt_d * scores` instead of
  `scores * self.inv_sqrt_d`, `args[0]` is the scalar node, not the
  matmul; the pattern silently fails to match. The error message points
  the user at "StandardAttention", not at the operand order.
- Risk: **Low**.

### L-2. Transpose matcher accepts arbitrary axes

- `_is_transpose` returns True for *any* `transpose` call (line 241-247).
  A user who accidentally writes `K.transpose(1, 2)` (not the last two
  dims) still passes the matcher; the kernel then receives K with the
  wrong layout and either crashes or produces wrong numbers.
- Risk: **Low** — a user who got that wrong would notice immediately,
  and typical code is `K.transpose(-2, -1)`.

### L-3. Injector constructor requires CUDA even for CPU-only tests

- `BlockSparseAttentionInjector.__init__` → `_build_mask` →
  `make_streaming_mask(... device=cfg.device)` with default
  `device="cuda"`. On a CPU-only machine the mask tensor would be
  allocated on a non-existent CUDA device at *construction* time, not
  at `.transform()` time. Easy fix: default `device="cpu"` and have the
  caller `.to("cuda")` the whole GraphModule.
- Risk: **Low**. Only matters for integration tests on CPU.

### Nothing else obviously catastrophic in a read-through

The forward kernel's online softmax follows FlashAttention-2 as
documented and matches the Dao 2023 formulas; the backward splitting
into dK/dV and dQ passes avoids `atomicAdd` on dQ exactly as the paper
prescribes; `D_i = <dO_i, O_i>` is pre-computed in its own tiny kernel.
Stride plumbing in the autograd wrapper looks internally consistent (all
four strides for Q/K/V/O, three for LSE/Delta, two for the block mask).
Nothing jumps out as a type mismatch (`acc` is fp32, cast to the output
dtype on store; `p` is cast to `v.dtype` before the P@V matmul).

The TWO structural correctness problems are H-1 and H-2; everything
else is either a robustness improvement or a reporting improvement.

## 5. What is needed to run on an NVIDIA laptop tomorrow

### OS

- Triton is Linux-only officially. If the laptop ships with Windows, use
  WSL2 with CUDA-on-WSL enabled; alternatively dual-boot into Linux.
- The macOS laptop this repo was authored on cannot run anything in
  `sparse_triton_kernel.py` beyond import-level checks.

### Toolchain

- Python ≥ 3.10 (set by `pyproject.toml`).
- `torch >= 2.3` and `triton >= 2.3` (per `requirements.txt`). Current
  stable at time of writing is torch 2.5 + triton 3.x; both should be
  fine. Prompt 003 bumps the floor to torch 2.5 for FlexAttention.
- Matching CUDA toolkit (whatever ships with the torch wheel; usually
  CUDA 12.x). Verify with `nvidia-smi` for driver + `python -c "import
  torch; print(torch.version.cuda)"` for the runtime CUDA the wheel was
  built against.
- GCC available for Triton's inline PTX / NVVM paths.

### GPU memory for default shapes

Default `HarnessConfig`: B=1, H=8, N_Q=1, N_KV=131072, D=64, fp16, mask
block = 64×64. Order-of-magnitude footprint of the **Triton path**:
- Q: `1·8·1·64·2` = 1 KiB
- K + V: `2 · 1·8·131072·64·2` = **256 MiB** (fp16)
- O, LSE: trivial
- Block mask: `1 · 2048 · 4` = 8 KiB

→ Triton path fits in <2 GiB with headroom for compilation workspace.

**Eager path** at the same shape materialises `QK^T` which is
`1·8·1·131072·4` = 4 MiB (fp32) for a *decode* step (N_Q=1), so eager
also fits. For `verify-fwd` the eager path operates on a scaled-down
256×256 and is trivial. For `verify-grads` the direct-allclose at
B=1 H=4 N=512 D=64 fp32 reaches `1·4·512·512·4` = 4 MiB — negligible.

The eager path only blows up when N_Q is pushed into the hundreds of
tokens (prefill rather than decode). The default is decode, so an 8 GiB
laptop GPU will run the defaults without pressure.

### Shapes to shrink on an 8 GiB laptop GPU

- `benchmark.py benchmark --batch 1 --heads 8 --seq-q 1 --seq-len 65536
  --head-dim 64 --dtype fp16`
  — halve the KV cache from the 131072 default to avoid any Triton
  compilation-time memory pressure and to let the eager path co-run for
  comparison. (The default of 131072 is fine for the Triton path alone;
  halving mostly matters when running eager side-by-side with prefill.)
- For a *prefill* smoke test:
  `--seq-q 512 --seq-len 512 --head-dim 64 --heads 4 --dtype fp16`
  — this is a shape where eager still fits (4·512·512·4 = 4 MiB) so the
  comparison is meaningful.
- `verify-grads` runs on hard-coded tiny shapes (64×64×16 and
  512×512×64); it does not scale with `--seq-len`. It will fit on an
  8 GiB card trivially.

### Known failure modes to look out for on first contact

- Triton kernel autotune cache lives under `.triton/`; first run will
  take 10-30 s per kernel compile. Not a bug, just slow.
- `tl.dot` on fp64 is likely unsupported on a laptop GPU; the fp64
  gradcheck will raise and silently drop to fp32 (M-3 above). Expected.
- If the laptop GPU is Ampere (SM80) or newer, `num_warps=4,
  num_stages=2` should be fine; on Ada (SM89) they are also fine but
  not tuned. If it's Turing (SM75), bf16 matmul is unsupported and the
  `--dtype bf16` CLI switch would fail — stick to fp16/fp32.

## 6. Index of what to change next (NOT changed in this prompt)

Prompt 001 explicitly says "diagnosis only — do NOT fix anything yet",
so this file ends here. Prompt 003 will fix H-1, H-2, M-2, M-3 and
widen the FX pass to match SDPA.
