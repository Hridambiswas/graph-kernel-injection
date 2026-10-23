# 003 report — Make it honest and GPU ready

Date: 2026-10-23. Branch: `feat/gpu-verification` (as required).
Prompt 001 and prompt 002 reports preceded this one. Every change
in this prompt landed in its own logical commit; `git log --oneline
main..HEAD` on `feat/gpu-verification` is the authoritative list.

Nothing in this prompt was executed on GPU — this host is still the
Mac. Every claim below is either (a) proven by CPU-runnable
`pytest`, (b) explicitly marked "fix written, GPU-unconfirmed", or
(c) a transparent documentation / tooling change.

## 1. HIGH-risk correctness fixes (prompt 003 step 1)

### H-1 — Causality is now enforced inside blocks

- Where: `sparse_triton_kernel.py:_fwd_kernel`,
  `_bwd_dkdv_kernel`, `_bwd_dq_kernel`.
- Each kernel now carries `IS_CAUSAL: tl.constexpr` and the runtime
  scalars `sink_size, window_size`. When `IS_CAUSAL` is true, every
  visited block applies the token-level mask
  `(k_pos < sink) | ((k_pos <= q_pos) & (k_pos >= q_pos - W + 1))`
  on top of the existing N bounds mask.
- The autograd `_BlockSparseFlashAttention.forward/backward` thread
  the new args through to the three kernel launches. The public
  `block_sparse_attention(..., is_causal=False, sink_size=0,
  window_size=0)` adds the three kwargs with sensible defaults so
  dense-block-mask callers see no behaviour change.
- Status: **fix written, GPU-unconfirmed**. The Python-level mask
  formula is proven equal to the paper's definition in
  `tests/test_mask_equivalence.py` (24 cases across 9 shape
  variants, all pass). Whether the Triton compiler lowers the new
  `IS_CAUSAL` branch correctly can only be confirmed by running
  `benchmark.py verify-grads` on the GPU host.

### H-2 — Oracle rebuilt from the paper's definition

- Where: `sparse_triton_kernel.py:reference_block_sparse_attention`.
- Oracle now takes the same `(is_causal, sink_size, window_size)`
  kwargs and intersects the lifted block mask with the token-level
  StreamingLLM mask — identical to what the kernel applies — so
  gradcheck no longer validates like-with-like against a model of
  attention that is not real StreamingLLM.
- New helper `streaming_token_mask(n_q, n_kv, sink, window)` returns
  the paper's token mask directly without routing through any block
  gating. `tests/test_oracle.py` compares the fixed oracle (at fp64
  on CPU) against a scratch dense computation using
  `streaming_token_mask`, over four shape variants, all pass. A
  companion regression test asserts that **turning off** the fix
  (`is_causal=False`) produces numerically different output —
  catching any future change that silently removes the fix.

### `benchmark.py` call sites updated

Every call to `block_sparse_attention` and
`reference_block_sparse_attention` in the harness now passes
`is_causal=True` with the configured sink / window — otherwise
`verify-grads` and `verify-fwd` would continue validating the old
wrong semantics even with the kernel fix in place.

### Deferred

- M-1 (mask constructor requires exact block divisibility, autograd
  bridge uses ceil division): no user-reachable crash because the
  mask builder shouts first; not addressed.
- M-2 (fully-masked row → `-inf` lse → NaN in bwd): a well-formed
  StreamingLLM mask with `sink_size >= 1` has no empty rows, so
  this cannot fire in the benchmarked configurations. Flagged in
  `docs/architecture_notes.md §4` and §7, not fixed here.
- M-3 (silent fp64→fp32 fallback in `verify-grads`): the log already
  announces the fallback; `benchmark.py` was not changed to also
  track which tier succeeded in the final summary. Flagged, not
  fixed.
- L-1, L-2, L-3: not addressed. Not blocking the GPU run.

## 2. Fair baselines (prompt 003 step 2)

Rewrote `benchmark.run_benchmark` to compare five methods on the
**same** token-level StreamingLLM mask:

| label                | backend                                                |
| :------------------- | :----------------------------------------------------- |
| `triton-block-sparse`| this repo's kernel (`is_causal=True` + sink + window)  |
| `flex-attention`     | `torch.nn.attention.flex_attention` with a block mask built from a `mask_mod` matching StreamingLLM. Requires torch ≥ 2.5. |
| `sdpa-explicit-mask` | `F.scaled_dot_product_attention(attn_mask=…)` with the identical `[N_Q, N_KV]` boolean mask |
| `sdpa-causal-dense`  | `F.scaled_dot_product_attention(is_causal=True)` — dense causal "no sparsity" reference |
| `eager-dense`        | `reference_block_sparse_attention` (Θ(N²) scratch)     |

Per-method report: fwd `p50/p95`, fwd+bwd `p50/p95`, peak VRAM,
max abs error vs an fp64 oracle at a scaled-down shape (so the
oracle actually fits). OOM / exception on any single method is
logged and the sweep continues with the others.

`requirements.txt`: `torch>=2.5` (for FlexAttention), `pytest>=7.0`
(for the CPU tests). `triton>=2.3` unchanged.

CSV header is now
`(label, dtype, seq_kv, mask_density, fwd_p50_ms, fwd_p95_ms,
fwdbwd_p50_ms, fwdbwd_p95_ms, peak_vram_MiB, max_abs_err_vs_fp64)`.

## 3. FX pass widened to SDPA (prompt 003 step 3)

- `fx_compiler_pass.py`: new `_match_sdpa_subgraph` walks the graph
  for `F.scaled_dot_product_attention` call_function nodes where
  `is_causal=True` and `dropout_p=0`. Each match is rewritten to a
  `_InjectedTritonAttention` call_module just like the eager path.
- Non-causal SDPA is **deliberately refused** with the standard
  `RuntimeError("No scaled-dot-product-attention subgraph found")`;
  the kernel can't absorb a dynamic `attn_mask` tensor at this stage.
- Install order: eager and SDPA matches collected first, then
  rewritten in **reverse topological order** so a graph with two
  chained SDPA calls doesn't leave stale node references in the
  match dataclass. Found via a failing CPU test; captured as a
  regression test.
- `tests/test_fx_sdpa.py`: four CPU-only cases — single causal
  SDPA rewritten, two chained causal SDPAs both rewritten,
  non-causal SDPA refused, installed child module carries the right
  `(is_causal, sink_size, window_size)` from the injector config.
  All pass.
- No HF or MHA tests exist. The README's §3 is explicit that neither
  is matched today.

## 4. `scripts/gpu_run.sh` (prompt 003 step 4)

Single command for tomorrow's run. Behaviours (verified by `bash -n`
syntax-check):

- Writes everything to `results/<UTC-timestamp>/`.
- `env.txt` captures `nvidia-smi`, python version, torch /
  torch.cuda / triton versions, GPU name + SM + total memory, and
  whether `flex_attention` imports.
- Runs `verify-grads`, `verify-fwd`, `benchmark` in order, teeing
  each stage's raw stdout to `01_verify_grads.log`,
  `02_verify_fwd.log`, `03_benchmark.log`.
- `set -euo pipefail` → stops at the first failure, leaves the log
  in place.
- `--small` flag selects shapes trimmed for an 8 GiB laptop GPU
  (heads=4, N_KV=16384, window=1024, warmup=10, iters=25); the
  full-size default mirrors `HarnessConfig`.
- Final step copies `bench_results.csv` and any `traces/` into the
  timestamped results dir so a run is self-contained.

## 5. `docs/eval_plan.md` (prompt 003 step 5)

Committed BEFORE any GPU run so results cannot shape the plan.
Contents:

- Three shapes (A: laptop decode, B: long-context decode, C: optional
  prefill smoke test) with every knob fixed.
- Timing protocol: `cuda.Event + synchronize`, 25/100 warmup/iters
  full-size, 10/25 on `--small`, peak VRAM via
  `max_memory_allocated()`.
- Correctness gate: `verify-grads` must pass before any perf number
  is reported; the fp64 → fp32 fallback is expected on consumer GPUs
  and the log line that confirms it must be present.
- Win / tie / loss definition vs FlexAttention (the only truly
  apples-to-apples baseline). Thresholds fixed in advance.
- Honesty disclaimer: ties and losses reported as plainly as wins;
  no HF / MHA / fp64-strict claims.

## 6. README honesty pass (prompt 003 step 6)

- New Status block at the top: "Not yet run on GPU", numeric claims
  flagged **analytical**, Hridam-reviewed / Hridam-verified fields
  left as TODO.
- Dropped "production-grade" from the opening paragraph.
- §1 memory-wall section: every figure explicitly labelled
  analytical; "translates almost linearly into decode speed"
  softened to "the actual multiplier is not yet measured".
- §3 FX pass: now covers both matchers and admits what is **not**
  matched (`MultiheadAttention`, HF modules). Dropped the false
  "idempotent" claim.
- §5 Correctness: states plainly that fp64 gradcheck typically
  cannot run on consumer GPUs and falls back to the FA2 fp32
  protocol; mentions the 28 CPU unit tests.
- §6 Benchmarks: deleted the placeholder table (its "**OOM**" eager
  cell was unverified) and replaced with a pointer to the eval plan
  and the five methods benchmarked.
- §7 Install & usage: adds `pytest tests/` as the CPU-only path and
  `bash scripts/gpu_run.sh [--small]` as the GPU-host wrapper.

## 7. Commits on `feat/gpu-verification` for this prompt

```
94f5a45 fix(kernel,oracle): apply token-level StreamingLLM mask inside blocks (H-1, H-2)
b9156d6 feat(bench): fair baselines (FlexAttention, SDPA mask, SDPA causal, eager)
4e0109d feat(fx): match F.scaled_dot_product_attention(is_causal=True)
5341765 feat(scripts): gpu_run.sh — one command for tomorrow
3393e02 docs: eval_plan.md — shapes/protocol/win criteria committed before any GPU run
f75abb9 docs(readme): status block + honesty pass per prompt 003 step 6
```

Plus this report, committed next on the same branch. Real
timestamps throughout; no `--date` overrides. Every commit ends
with the Co-Authored-By line for Claude. Not pushed (no remote).

## 8. Final state checklist

- [x] 28 CPU unit tests pass on this host (`pytest tests/`).
- [x] `bash -n scripts/gpu_run.sh` exits 0.
- [x] On `feat/gpu-verification` throughout; `main` untouched since
      prompt 002's single root commit.
- [x] `archive/staged-history` still exists (local-only backup from
      prompt 002).
- [x] `docs/next_session.md` still carries the local-only warning.
- [x] `docs/eval_plan.md` committed before any GPU run.
- [x] No pushes to any remote; still no remote configured.
- [x] No commits on `main` since the honest root commit.

## 9. Open questions and known blind spots

- **GPU codegen for the new `IS_CAUSAL` branch is unverified.** The
  CPU tests prove the mask arithmetic matches the paper; the Triton
  codegen can only be confirmed by `verify-grads` on hardware. If
  the kernel's constexpr `IS_CAUSAL` path has a Triton-level bug
  (e.g. a wrong broadcast on the sink mask), the CPU tests do not
  catch it.
- **FlexAttention `mask_mod` compile cost** at the full-size shape
  (N_KV = 131 072) can be substantial. The eval plan uses 25 warmup
  iters which should amortise it, but if the FlexAttention warmup
  dwarfs its steady-state latency the comparison needs re-tuning;
  prompt 004 can address this if it comes up.
- The `sdpa-causal-dense` baseline is **not** a StreamingLLM
  baseline — it is dense causal, not sparse. It is included only to
  show what the FlashAttention backend costs at the configured
  shape. We will not claim a "win" against it.
- **No HF / MHA integration test.** Only the hand-written
  `StandardAttention` and `F.scaled_dot_product_attention(is_causal=
  True)` modules have tests. HF compatibility would need prompt
  004+ to deliver.
