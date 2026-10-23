# 004 report — Correctness fixes to a clean stopping point

Date: 2026-10-23. Branch: `feat/gpu-verification`. Host: Mac, CPU
only (no Triton, no CUDA). No benchmark was executed — this prompt
is strictly correctness and honesty cleanup.

## 1. Two critical mask bugs (director review, prompt 004 §1)

The CPU tests from prompt 003 passed only because
`reference_block_sparse_attention`, `streaming_token_mask` and the
Triton kernel all shared the same wrong semantics (an H-2-class
problem again). Both bugs were fixed at every site and a new
hand-written reference was added that calls no repo helper.

### A. Sink leaked the future

- **Where** (all five sites): the kernel at `sparse_triton_kernel.py`
  (`_fwd_kernel` ~line 296, `_bwd_dkdv_kernel` ~line 457,
  `_bwd_dq_kernel` ~line 559), the oracle in
  `reference_block_sparse_attention` ~line 848, and `streaming_token_mask`
  ~line 887. Also the FlexAttention `mask_mod` in `benchmark.py::_flex_fn`.
- **Old formula**: `token_mask = sink | (causal & window)`.
  For a query at position `q < sink_size`, the `sink` branch
  unconditionally included every `k < sink_size`, so `q=0` could
  attend to `k=1, 2, 3`. The oracle, the token-mask helper and the
  FlexAttention `mask_mod` all had this bug too — gradcheck
  compared like-with-like against the wrong semantics.
- **New formula**: `causal & (sink | window)`. Every attended cell
  must satisfy `k <= q`; within the causal half-plane, the key is
  attended iff it is in the sink prefix or inside the sliding
  window.
- **Docstring update**: public API `block_sparse_attention(..., is_causal, …)`
  carries the correct formula; the kernel bodies have inline
  comments naming prompt 004 bug A.

### B. Query positions ignored the KV offset

- **Where**: kernel (all three kernels), oracle, `streaming_token_mask`,
  `make_streaming_mask`, FlexAttention `mask_mod`, and the SDPA
  explicit-mask baseline (which routes through `streaming_token_mask`).
- **Old convention**: `q_pos = arange(N_Q)` — the first query sat at
  position 0.  For decode with `N_Q=1, N_KV=131072` the lone query
  was treated as token 0 and could see only the sink prefix — a few
  tokens — which is both wrong and would have faked a huge speedup
  on the benchmark.
- **New convention**: `q_pos = arange(N_KV - N_Q, N_KV)`, i.e. the
  bottom-right alignment FlashAttention and FlexAttention use with
  a KV cache. The kernel takes a new runtime scalar `q_offset =
  N_KV - N_Q`; `make_streaming_mask` shifts its block-level
  `q_lo / q_hi` coordinates by `q_offset`; the autograd bridge
  computes `q_offset` from the input shapes and threads it through
  the three kernel launches.
- **Decode sanity**: for `N_Q=1, N_KV=1000, S=4, W=256` (no
  overlap) the lone query at position 999 must see `S + W = 260`
  keys, not ~`S`. A unit test asserts this.

### C. Independent definition-based tests

New file `tests/test_streaming_definition.py` implements the paper's
definition **by hand**, calling no repo helper inside the reference:

```python
def _visible(q_abs, k_abs, sink, window):
    if k_abs > q_abs:        return False  # causal
    if k_abs < sink:          return True   # sink
    if q_abs - k_abs < window: return True   # window
    return False
```

`i = (N_KV - N_Q) + row`. The tests check every mask producer
against this reference across these shape families:

- **N_Q < S**: `(4, 10, 8, 4)` — queries live inside the sink prefix.
- **non-block-divisible**: `(1000, 1000)`, `(1023, 1023)`, `(1025, 1025)`
  with `bm=bn=64`, plus `(1, 1023)`, `(1, 1025)` with `bm=1, bn=64`.
- **decode** `N_Q=1, N_KV=1000`: visible-count sanity (`S+W`),
  window-only, sink-only, and window-covers-all.
- **asymmetric**: `(16, 1000)`, `(128, 1024)`.

The existing `test_block_mask_is_superset_of_token_mask` test is
retained (now in both the old and new files; the new file adds
non-divisible shapes).

Direct oracle output is also checked against a hand-built dense
`softmax(QK^T + hand_mask) @ V` at fp64, so a shared bug in the
oracle and the token-mask helper cannot hide.

## 2. Cheap medium fixes (prompt 004 §2)

### M-1 — ceiling division

`make_streaming_mask` previously raised when `n_q % block_m != 0` or
`n_kv % block_n != 0`. The autograd bridge used ceil division, so the
two helpers disagreed on edge-block shapes. The mask builder now also
uses `(n + b - 1) // b`, partial edge blocks are retained, and the
kernel's existing bounds checks (`offs_m < N_Q`, `kv_mask`) handle
the token-level trimming. The block mask shape is now exactly
`(ceil(n_q / block_m), ceil(n_kv / block_n))` — matching the autograd
bridge's expectation.

### M-2 — fully-masked rows produce zero, never NaN

A query row whose block mask gates every key hits `l_i == 0` and
`m_i == -inf` in the forward kernel. Previously the stored LSE was
`m_i + log(1.0) = -inf`; the backward then evaluated
`exp(-inf - -inf) = exp(NaN) = NaN` and poisoned `dQ, dK, dV`.

Fix in `_fwd_kernel`:

```
row_valid = l_i > 0.0
safe_l = tl.where(row_valid, l_i, 1.0)
safe_m = tl.where(row_valid, m_i, 0.0)        # finite sentinel
acc    = acc / safe_l[:, None]                 # 0/1 = 0 on empty rows
lse    = safe_m + tl.log(safe_l)               # 0 + log(1) = 0 on empty
```

Downstream in the backward, `p = exp(qk - lse) = exp(-inf - 0) = 0`
for every cell in an empty row → zero contribution to `dK/dV` and
`dQ`, no NaN.

GPU-only test: `tests/test_fully_masked_row.py` builds a custom block
mask that zeros out the first query block, runs forward+backward,
and asserts (a) the output is finite everywhere, (b) rows 0..`BLOCK_M-1`
are exactly zero in both output and `dQ`. The file top is guarded
with `pytest.importorskip("triton")` and
`torch.cuda.is_available()` so CPU hosts skip it (1 skipped on this
Mac host; the file will run on the GPU host).

### M-3 — gradcheck fp64/fp32 reporting

`benchmark.py::verify_gradients` now:
- Tracks `fp64_ran: bool` and `fp64_skip_reason: Optional[str]`.
- Prints the mandated line **`FP64 GRADCHECK NOT RUN: <reason>`** on
  fallback (verbatim, as 004 §2 M-3 requires).
- Lets a fp32-fallback failure propagate uncaught, so the CLI exits
  non-zero on any gradcheck failure. The intermediate `_run_gradchecks`
  call is deliberately NOT wrapped in a second try/except.
- Ends with `[summary] gradcheck.fp64_ran=<bool>  (fp64 skip reason: …)`
  so the final summary records which tier ran, as required.

### L-3 — lazy mask on CPU

`BlockSparseAttentionInjector._build_mask` now always allocates on
CPU regardless of `cfg.device`. The block mask is registered as a
non-persistent buffer on `_InjectedTritonAttention`, which follows
`.to(device)` propagation when the user calls `fused.to("cuda")`
after `.transform(...)`. Previously the default `cfg.device="cuda"`
blew up at construction time on a CPU-only host, defeating FX
tracing.

## 3. Housekeeping (prompt 004 §3)

- `.gitignore` gained:
  ```
  # claude code harness
  .claude/settings.local.json
  .claude/scheduled_tasks.lock
  ```
- `README.md` Status block: now states "Kernel and FX pass
  implemented; **mask semantics verified by CPU tests** (58 cases
  under `tests/`, including a hand-written reference that calls no
  repo helper so a shared bug cannot hide). **NOT yet run on GPU,
  no performance numbers yet; GPU verification planned.**" The AI
  assistance line and the `Hridam reviewed / Hridam verified: TODO`
  fields are retained. §5 Correctness mentions the new `FP64
  GRADCHECK NOT RUN` log line and the updated test count (58).
- `docs/next_session.md`: GPU run is called out as the next real
  milestone; the exact commands (`bash scripts/gpu_run.sh` and
  `bash scripts/gpu_run.sh --small`) are listed. 005 is called out
  as the pre-approved next prompt in the inbox.

## 4. Test output

Full output of `python -m pytest tests/ -v` on this Mac (no Triton,
no CUDA — the GPU-only `test_fully_masked_row.py` is skipped):

```
============================= test session starts ==============================
platform darwin -- Python 3.10.11, pytest-9.0.3, pluggy-1.6.0 -- /Users/hridambiswas/.pyenv/versions/3.10.11/bin/python
cachedir: .pytest_cache
rootdir: /Users/hridambiswas/graph-kernel-injection
configfile: pyproject.toml
plugins: anyio-4.12.1, langsmith-0.7.38, cov-7.1.0, asyncio-1.3.0
asyncio: mode=strict, debug=False, asyncio_default_fixture_loop_scope=None, asyncio_default_test_loop_scope=function
collecting ... collected 58 items / 1 skipped

tests/test_fx_sdpa.py::test_fx_rewrites_single_causal_sdpa_call PASSED   [  1%]
tests/test_fx_sdpa.py::test_fx_rewrites_multiple_causal_sdpa_calls PASSED [  3%]
tests/test_fx_sdpa.py::test_fx_refuses_non_causal_sdpa PASSED            [  5%]
tests/test_fx_sdpa.py::test_fx_injected_module_carries_streamingllm_params PASSED [  6%]
tests/test_mask_equivalence.py::test_kernel_mask_matches_streamingllm_definition[16-16-4-8-8-8] PASSED [  8%]
tests/test_mask_equivalence.py::test_kernel_mask_matches_streamingllm_definition[32-32-4-8-8-8] PASSED [ 10%]
tests/test_mask_equivalence.py::test_kernel_mask_matches_streamingllm_definition[64-64-4-16-16-16] PASSED [ 12%]
tests/test_mask_equivalence.py::test_kernel_mask_matches_streamingllm_definition[64-128-4-32-16-32] PASSED [ 13%]
tests/test_mask_equivalence.py::test_kernel_mask_matches_streamingllm_definition[128-128-8-32-32-32] PASSED [ 15%]
tests/test_mask_equivalence.py::test_kernel_mask_matches_streamingllm_definition[256-256-4-64-64-64] PASSED [ 17%]
tests/test_mask_equivalence.py::test_kernel_mask_matches_streamingllm_definition[128-1024-4-128-64-64] PASSED [ 18%]
tests/test_mask_equivalence.py::test_kernel_mask_matches_streamingllm_definition[64-64-0-64-16-16] PASSED [ 20%]
tests/test_mask_equivalence.py::test_kernel_mask_matches_streamingllm_definition[64-64-64-0-16-16] PASSED [ 22%]
tests/test_mask_equivalence.py::test_block_mask_is_superset_of_token_mask[16-16-4-8-8-8] PASSED [ 24%]
tests/test_mask_equivalence.py::test_block_mask_is_superset_of_token_mask[32-32-4-8-8-8] PASSED [ 25%]
tests/test_mask_equivalence.py::test_block_mask_is_superset_of_token_mask[64-64-4-16-16-16] PASSED [ 27%]
tests/test_mask_equivalence.py::test_block_mask_is_superset_of_token_mask[64-128-4-32-16-32] PASSED [ 29%]
tests/test_mask_equivalence.py::test_block_mask_is_superset_of_token_mask[128-128-8-32-32-32] PASSED [ 31%]
tests/test_mask_equivalence.py::test_block_mask_is_superset_of_token_mask[256-256-4-64-64-64] PASSED [ 32%]
tests/test_mask_equivalence.py::test_block_mask_is_superset_of_token_mask[128-1024-4-128-64-64] PASSED [ 34%]
tests/test_mask_equivalence.py::test_block_mask_is_superset_of_token_mask[64-64-0-64-16-16] PASSED [ 36%]
tests/test_mask_equivalence.py::test_block_mask_is_superset_of_token_mask[64-64-64-0-16-16] PASSED [ 37%]
tests/test_mask_equivalence.py::test_pre_fix_bug_is_visible_without_in_block_mask PASSED [ 39%]
tests/test_oracle.py::test_oracle_matches_streamingllm_paper[32-32-4-8-8-8] PASSED [ 41%]
tests/test_oracle.py::test_oracle_matches_streamingllm_paper[64-64-4-16-16-16] PASSED [ 43%]
tests/test_oracle.py::test_oracle_matches_streamingllm_paper[64-128-4-32-16-32] PASSED [ 44%]
tests/test_oracle.py::test_oracle_matches_streamingllm_paper[128-128-8-32-32-32] PASSED [ 46%]
tests/test_oracle.py::test_oracle_pre_fix_leaks_the_bug PASSED           [ 48%]
tests/test_streaming_definition.py::test_streaming_token_mask_matches_hand_definition[1-1000-4-256] PASSED [ 50%]
tests/test_streaming_definition.py::test_streaming_token_mask_matches_hand_definition[1-1000-0-256] PASSED [ 51%]
tests/test_streaming_definition.py::test_streaming_token_mask_matches_hand_definition[1-1000-4-0] PASSED [ 53%]
tests/test_streaming_definition.py::test_streaming_token_mask_matches_hand_definition[1-1000-4-2048] PASSED [ 55%]
tests/test_streaming_definition.py::test_streaming_token_mask_matches_hand_definition[16-16-4-8] PASSED [ 56%]
tests/test_streaming_definition.py::test_streaming_token_mask_matches_hand_definition[64-64-4-16] PASSED [ 58%]
tests/test_streaming_definition.py::test_streaming_token_mask_matches_hand_definition[2-10-4-4] PASSED [ 60%]
tests/test_streaming_definition.py::test_streaming_token_mask_matches_hand_definition[1000-1000-4-128] PASSED [ 62%]
tests/test_streaming_definition.py::test_streaming_token_mask_matches_hand_definition[1023-1023-4-128] PASSED [ 63%]
tests/test_streaming_definition.py::test_streaming_token_mask_matches_hand_definition[1025-1025-4-128] PASSED [ 65%]
tests/test_streaming_definition.py::test_streaming_token_mask_matches_hand_definition[4-10-8-4] PASSED [ 67%]
tests/test_streaming_definition.py::test_streaming_token_mask_matches_hand_definition[16-1000-4-128] PASSED [ 68%]
tests/test_streaming_definition.py::test_streaming_token_mask_matches_hand_definition[128-1024-4-128] PASSED [ 70%]
tests/test_streaming_definition.py::test_decode_sees_sink_plus_window_tokens PASSED [ 72%]
tests/test_streaming_definition.py::test_sink_does_not_leak_the_future PASSED [ 74%]
tests/test_streaming_definition.py::test_block_mask_is_superset_of_hand_token_mask[1-1000-4-256-1-64] PASSED [ 75%]
tests/test_streaming_definition.py::test_block_mask_is_superset_of_hand_token_mask[1-1023-4-256-1-64] PASSED [ 77%]
tests/test_streaming_definition.py::test_block_mask_is_superset_of_hand_token_mask[1-1025-4-256-1-64] PASSED [ 79%]
tests/test_streaming_definition.py::test_block_mask_is_superset_of_hand_token_mask[16-1000-4-128-16-64] PASSED [ 81%]
tests/test_streaming_definition.py::test_block_mask_is_superset_of_hand_token_mask[1000-1000-4-128-64-64] PASSED [ 82%]
tests/test_streaming_definition.py::test_block_mask_is_superset_of_hand_token_mask[1023-1023-4-128-64-64] PASSED [ 84%]
tests/test_streaming_definition.py::test_block_mask_is_superset_of_hand_token_mask[1025-1025-4-128-64-64] PASSED [ 86%]
tests/test_streaming_definition.py::test_block_mask_is_superset_of_hand_token_mask[4-10-8-4-2-2] PASSED [ 87%]
tests/test_streaming_definition.py::test_block_mask_is_superset_of_hand_token_mask[16-16-4-8-8-8] PASSED [ 89%]
tests/test_streaming_definition.py::test_block_mask_is_superset_of_hand_token_mask[128-1024-4-128-16-64] PASSED [ 91%]
tests/test_streaming_definition.py::test_oracle_output_matches_hand_dense_computation[16-16-4-8-8-8] PASSED [ 93%]
tests/test_streaming_definition.py::test_oracle_output_matches_hand_dense_computation[32-32-4-8-8-8] PASSED [ 94%]
tests/test_streaming_definition.py::test_oracle_output_matches_hand_dense_computation[4-32-4-8-2-8] PASSED [ 96%]
tests/test_streaming_definition.py::test_oracle_output_matches_hand_dense_computation[1-32-4-8-1-8] PASSED [ 98%]
tests/test_streaming_definition.py::test_oracle_output_matches_hand_dense_computation[16-1000-4-128-16-64] PASSED [100%]

======================== 58 passed, 1 skipped in 5.84s =========================
```

28 cases from prompt 003 are still there plus 30 new cases from
`test_streaming_definition.py`. The one skipped test is
`tests/test_fully_masked_row.py` (M-2 regression; GPU-only).

## 5. What is still unverified until the GPU run

- Triton codegen for the new `q_offset` runtime arg and the swapped
  mask formula is proven only by reading. The CPU tests cover the
  Python-level mask arithmetic; the actual kernel bodies can only
  be validated by `benchmark.py verify-grads` on hardware.
- M-2 (fully-masked row → zero) is unit-tested on the GPU path but
  not run on this Mac. The test is in place and will fire when the
  GPU host runs `pytest tests/`.
- The FlexAttention `mask_mod` fix (bottom-right alignment plus
  the swapped formula) is only run inside the benchmark, which
  still requires GPU.
- M-3 fp64 fallback: the new `FP64 GRADCHECK NOT RUN: ...` log
  line and the final `[summary] gradcheck.fp64_ran=…` line will
  be observable in `02_verify_grads.log` on the GPU host (or
  wherever `benchmark.py verify-grads` runs).

## 6. Commits on `feat/gpu-verification` for this prompt

To be committed next on the same branch, with the Co-Authored-By
line for Claude. Real timestamps; no `--date` overrides.

## 7. Open questions

None. Prompt 005 (publish the public GitHub repo) is pre-approved
in its own text; the mailbox loop will pick it up on the next
cycle.
