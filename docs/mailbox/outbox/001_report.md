# 001 report — Read, map and audit

Date: 2026-10-23. Branch: `main` (prompt 001 explicitly authorises committing
on main because there is no remote and prompt 002 rebuilds history from
scratch). No code changed; this prompt is diagnosis only.

## 1. What I did

1. Confirmed git state:
   - `git branch --show-current` → `main`
   - `git remote -v` → empty (no remote)
   - `git log --format='%h %ad %s' --date=iso | head -80` → 68 commits
     spaced at exact 12-minute intervals from 2026-10-22 09:00 to 22:30
     IST. The messages describe a plausible but fictional development arc.
     This is the staged history prompt 002 is slated to replace.
2. Read every file in full:
   - `sparse_triton_kernel.py` (780 lines) — mask builders, Triton
     fwd + bwd kernels, autograd bridge, reference oracle.
   - `fx_compiler_pass.py` (490 lines) — pattern matcher + rewriter.
   - `benchmark.py` (660 lines) — correctness + perf harness.
   - `README.md`, `pyproject.toml`, `requirements.txt`, `.gitignore`.
3. Wrote `docs/architecture_notes.md`. That file is the primary artefact
   for this prompt; this report is a short summary of what it contains
   plus explicit answers to the prompt's bullet list.

Commands run (shell only, no code changed):

```
git branch --show-current
git remote -v
git log --format='%h %ad %s' --date=iso | head -80
git branch -a
ls -la /Users/hridambiswas/graph-kernel-injection/
cat /Users/hridambiswas/graph-kernel-injection/.gitignore
ls -la /Users/hridambiswas/graph-kernel-injection/.claude/
ls -la /Users/hridambiswas/graph-kernel-injection/results/
```

## 2. Architecture summary (full detail in `docs/architecture_notes.md`)

- **`sparse_triton_kernel.py`**: four Triton kernels (`_fwd_kernel`,
  `_bwd_preprocess_kernel`, `_bwd_dkdv_kernel`, `_bwd_dq_kernel`), an
  `autograd.Function` bridge, a `block_sparse_attention` public entry
  point, and `reference_block_sparse_attention` (dense Θ(N²) oracle).
  Mask builders are `make_streaming_mask` (block-level StreamingLLM
  mask) and `make_dense_mask`.
- **`fx_compiler_pass.py`**: `BlockSparseAttentionInjector` is a
  pattern-matcher + rewriter + lint pass that pivots on `softmax` nodes
  in a symbolic-traced graph and climbs back through
  `softmax → mul/truediv → matmul → transpose`, forward through
  `softmax → matmul(P, V)`. Found matches get replaced with a single
  `call_module` → `_InjectedTritonAttention`.
- **`benchmark.py`**: CLI harness with `verify-grads`, `verify-fwd`,
  `benchmark` modes. `verify_gradients` runs gradcheck +
  gradgradcheck + direct allclose; `main()` gates the perf benchmark
  behind both correctness passes.

## 3. What the FX pass matches today — blunt answer

The matcher locates only the repo's own hand-written
`softmax(matmul(scale(matmul(Q, K.transpose(-2, -1)))), dim=-1) @ V`
motif, as inlined into any parent `nn.Module`. It also matches small
syntactic variants (scale as literal / `get_attr` / `truediv`, softmax
as method / function / `F.softmax`, no scale at all).

It does **NOT** match:

- `torch.nn.functional.scaled_dot_product_attention` — SDPA is a
  single opaque `call_function` node; there is no softmax or inner
  matmul to pivot on. No test exists.
- `torch.nn.MultiheadAttention` — current PyTorch routes it through
  SDPA, and even if that is bypassed the module's `forward` has
  data-dependent branches that `symbolic_trace` cannot handle cleanly.
  No test exists.
- Hugging Face GPT-2 / Llama attention. Not in deps, not tested, and
  the pattern is broken by the surrounding dropout / RoPE / causal-mask
  additions that HF layers use. No test exists.
- `scale * scores` (operator order reversed) — the matcher looks for
  the matmul at `args[0]` of the scale node.
- `softmax(dim=k)` where `k != -1` even if numerically equivalent.
- `K.T` / `K.permute(0, 1, 3, 2)` instead of
  `K.transpose(-2, -1)` (depends on how FX lowers each).

**Current reality:** the pass is a demonstration that the plumbing
works on the repo's own module. It is not yet shown to be useful on
a real model, and the README does not quite admit that.

## 4. Correctness audit (reading only — no execution possible on this Mac)

Priorities and risk levels (full detail in `docs/architecture_notes.md §4`):

| ID   | Severity | Where                                                       | What                                                                                                                                                                                                 |
| :--- | :------- | :---------------------------------------------------------- | :--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| H-1  | **High** | `_fwd_kernel:285-287`, `_bwd_dkdv_kernel:430-431`, `_bwd_dq_kernel:523-524` | Causal mask is applied only at block granularity. On diagonal / near-diagonal blocks, token positions `k > q` are let through. Real StreamingLLM is strictly causal. **Bug.**                        |
| H-2  | **High** | `reference_block_sparse_attention:762-771`                  | The oracle derives its token mask by `repeat_interleave` on the block mask, so it shares the H-1 bug. `gradcheck` therefore compares like-with-like and will pass even though the kernel is wrong.    |
| M-1  | Medium   | `make_streaming_mask:141-144` vs `_BlockSparseFlashAttention.forward:573-574` | Mask builder requires exact block-divisibility; autograd bridge uses ceiling division. Only one of the two models non-multiple shapes.                                                               |
| M-2  | Medium   | `_fwd_kernel:300-312`                                       | Fully-masked rows store `lse = -inf`. `exp(-inf - (-inf)) = NaN` in bwd → NaN gradients. Fine for well-formed StreamingLLM (`sink_size >= 1`), poisonous for custom masks with empty rows.           |
| M-3  | Medium   | `benchmark.py:287-296`                                      | `verify-grads` wraps the fp64 attempt in `try/except Exception`, so on consumer GPUs (where `tl.dot` fp64 is unsupported) the harness silently falls back to fp32. Message is printed but exit is 0. |
| L-1  | Low      | `fx_compiler_pass.py:339`                                   | Matcher reads `qk_candidate = scaled_input.args[0]`, so `scale * scores` is not matched.                                                                                                             |
| L-2  | Low      | `fx_compiler_pass.py:241-247`                               | Transpose check accepts arbitrary axes, not just `(-2, -1)`.                                                                                                                                         |
| L-3  | Low      | `fx_compiler_pass.py:280-298`                               | `BlockSparseAttentionInjector.__init__` builds the mask on `cfg.device` (default `cuda`) at construction time; a CPU-only host crashes immediately.                                                 |

**No fixes applied** (per prompt 001). Prompt 003 will cover H-1, H-2,
M-2, M-3 and SDPA matching.

### Specific answers to the audit bullets in prompt 001

- *Bounds masks on loads/stores when N is not a multiple of block_m /
  block_n:* `_fwd_kernel` guards Q loads with `q_mask`, KV loads with
  `kv_mask`, O/LSE stores with the same masks. The dense-outside-mask
  bytes read as `other=0.0`. **Correct in principle, but the mask
  builder forbids non-multiple shapes in the first place (M-1), so this
  branch is never exercised.** No crash expected.
- *Fully-masked rows:* `safe_l = tl.where(l_i > 0.0, l_i, 1.0)` saves
  the division but not the `log`. See M-2. **Risk only materialises
  with custom masks.**
- *Causality inside blocks:* **not applied.** See H-1. **High-risk
  correctness bug.**
- *Oracle uses same token-level mask as kernel:* yes, but *because
  both derive from the same block mask* and *neither implements the
  real StreamingLLM token mask*. See H-2.
- *Does gradcheck run Triton in fp64 for real, or silently fall back:*
  **silently falls back** on any consumer GPU. See M-3. The log line
  announces it but the exit code does not.
- *Anything else that would crash:* nothing jumps out beyond the above.
  Stride plumbing is internally consistent; dtypes are mixed-precision
  the way FA2 prescribes; the three-pass backward matches the paper.

## 5. What is needed to run on an NVIDIA laptop tomorrow

Full detail in `docs/architecture_notes.md §5`. Headline items:

- **OS**: Linux (direct install or WSL2). Not Windows native, not macOS.
- **Toolchain**: Python ≥ 3.10, `torch >= 2.3`, `triton >= 2.3`,
  matching CUDA toolkit from the torch wheel.
- **GPU memory**: Default harness (B=1, H=8, N_Q=1, N_KV=131072, D=64,
  fp16) uses <2 GiB on the Triton path. Even the direct-allclose test
  at 512×512×D64 fp32 is <5 MiB. An 8 GiB laptop GPU is fine for the
  defaults.
- **Shrink for 8 GiB**: nothing *needs* shrinking at the defaults, but
  for a prefill smoke test keep `seq-q ≤ 1024, seq-kv ≤ 1024`.
- **Expected quirks**: first-run Triton compile takes 10-30 s per
  kernel; fp64 gradcheck will almost certainly fall back to fp32 on a
  laptop GPU (not a bug per se — see M-3).

## 6. Open questions / ambiguities

None that would block prompts 002 and 003. If I were to flag one for
Hridam: the README's claim "the pass is *idempotent*" (line 100-101) is
false in the current implementation — `.transform` raises
`RuntimeError("No scaled-dot-product-attention subgraph found. …")`
when the pattern is absent. Prompt 003 should decide whether to make
the pass genuinely idempotent (return the module unchanged) or whether
to delete that claim from the README.

## 7. Commit plan for this prompt

Per the prompt's final paragraph, this report commits:

- `docs/architecture_notes.md` (new, written by this prompt)
- `docs/mailbox/outbox/001_report.md` (this file)
- `CLAUDE.md`, `docs/next_session.md`, `.claude/settings.json`,
  `docs/mailbox/inbox/*.md` (previously untracked; pre-existing
  director artefacts, not touched by me beyond staging).

Normal current timestamp. On `main`, as the prompt explicitly
authorises (no remote, 002 rebuilds). Not pushed.
