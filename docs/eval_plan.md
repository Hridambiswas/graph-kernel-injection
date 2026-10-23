# Evaluation plan

Committed **before** the first GPU run so results can't shape the plan.
Prompt 003 step 5. Any deviation at run time must be noted in the
report under `docs/mailbox/outbox/`, not silently applied.

## 1. Hardware

The first run will be on Hridam's NVIDIA laptop (precise model
captured by `scripts/gpu_run.sh` into `results/<timestamp>/env.txt`).
Expected: single consumer GPU with 6–16 GiB HBM, Ampere or Ada
generation.

## 2. Shapes

### Shape A — "laptop decode" (default on `--small`)

Mirrors a KV-cache decode step at modest context length.

| field       | value   |
| :---------- | :------ |
| batch       | 1       |
| heads       | 4       |
| N_Q         | 1       |
| N_KV        | 16 384  |
| head_dim    | 64      |
| sink_size   | 4       |
| window_size | 1 024   |
| block_m     | 1       |
| block_n     | 64      |
| dtype       | fp16    |

Rationale: an 8 GiB laptop GPU fits this comfortably. All five
baselines (triton, flex, sdpa-mask, sdpa-causal, eager) should run.

### Shape B — "long-context decode" (full default)

The headline shape the README talks about.

| field       | value     |
| :---------- | :-------- |
| batch       | 1         |
| heads       | 8         |
| N_Q         | 1         |
| N_KV        | 131 072   |
| head_dim    | 64        |
| sink_size   | 4         |
| window_size | 2 048     |
| block_m     | 1         |
| block_n     | 64        |
| dtype       | fp16      |

Rationale: this is where the StreamingLLM mask has the biggest edge
(≈ 49× bandwidth reduction vs dense FlashAttention, analytically).
Eager dense is expected to run but is slow; FlexAttention and SDPA
should both still fit.

### Shape C — "prefill smoke test" (optional, run only if GPU memory
allows)

Not driven by `gpu_run.sh` directly; run manually only if B/A succeed
and the laptop has headroom. Lets us check the kernel on a non-decode
shape.

| field       | value |
| :---------- | :---- |
| batch       | 1     |
| heads       | 4     |
| N_Q         | 512   |
| N_KV        | 512   |
| head_dim    | 64    |
| sink_size   | 4     |
| window_size | 128   |
| block_m     | 64    |
| block_n     | 64    |
| dtype       | fp16  |

## 3. Timing protocol

- Warmup iterations: **25** (full-size), **10** (`--small`).
  Triton's first-run compile is 10–30 s per kernel; warmup gives the
  JIT cache time to populate before any sample lands in the latency
  histogram.
- Timed iterations: **100** (full-size), **25** (`--small`).
- Measurement path: `torch.cuda.Event(enable_timing=True)` around a
  single call plus a `torch.cuda.synchronize()` after each event, as
  already implemented in `benchmark.py:_measure` and
  `benchmark.py:_measure_fwdbwd`.
- Peak VRAM: `torch.cuda.max_memory_allocated()` after
  `reset_peak_memory_stats()` at the start of each method's timed
  window. Reported as the max over the fwd-only and fwd+bwd windows.

## 4. Correctness protocol

Before any perf numbers are reported for a run:

1. `benchmark.py verify-grads` must pass. Expected on a consumer GPU:
   the fp64 path raises an unsupported-dtype error at `tl.dot`; the
   harness falls back to the FA2 fp32 protocol (`eps=1e-3,
   atol=5e-3, rtol=5e-3`) and must pass that. The log line
   `[gradcheck] ✓ fp32 FA2 fallback completed` must be present. If
   the fp64 path **does** succeed, note that in the report — it means
   the GPU is unexpectedly datacenter-class.
2. `benchmark.py verify-fwd` must pass. The 256×256 fp32 and fp16
   allclose checks must both report `allclose=True`.

If either step fails, the perf benchmark is **not** reported. The
failing log is kept under `results/<timestamp>/` and the next prompt
investigates.

## 5. What counts as a win, tie, or loss vs FlexAttention

The one directly-comparable baseline is FlexAttention: it builds the
same block-level mask from the same `mask_mod`, uses Triton under the
hood, and targets the same arithmetic. Comparisons:

- **Win** at this shape: `triton-block-sparse.fwd_p95_ms` is more
  than **1.2 ×** faster than `flex-attention.fwd_p95_ms`, **and**
  `max_abs_err_vs_fp64` of the Triton path is no worse than
  2 × that of FlexAttention at the same dtype.
- **Tie**: the ratio is within [0.9, 1.1], OR either method's
  variance (`p95/p50 > 2`) makes the comparison unreliable.
- **Loss**: `triton-block-sparse.fwd_p95_ms` is more than **1.1 ×**
  slower than FlexAttention OR the max abs error is more than 2 ×
  worse at the same dtype.

Honesty rule: ties and losses are reported as plainly as wins. The
README's final numbers table includes every method in the sweep,
not only the ones where this kernel comes out ahead.

For the comparisons that are *not* apples-to-apples (SDPA causal is
dense; eager is Θ(N²)), we report them for context but no
"win/loss" verdict is attached — those methods implement different
semantics and the comparison is only meaningful as "what does
dense cost at this shape".

## 6. What goes into `results/<timestamp>/`

- `env.txt` — hardware / toolchain capture.
- `01_verify_grads.log`, `02_verify_fwd.log`, `03_benchmark.log` —
  raw stdout per stage. Never edited after they're written (CLAUDE.md
  honesty rule).
- `bench_results.csv` — one row per method.
- `traces/triton.json` + `traces/eager.json` — Chrome traces, if the
  benchmark generated them.

The whole `results/<timestamp>/` tree is read-only once the run
exits.

## 7. What we are *not* claiming from this run

- Hugging Face GPT-2 / Llama compatibility. The FX pass is only
  tested on hand-written `StandardAttention` and on a module calling
  `F.scaled_dot_product_attention(is_causal=True)`. No HF-model
  tracing test exists.
- `torch.nn.MultiheadAttention` compatibility. Not tested.
- fp64 strict gradcheck on consumer GPUs. Documented fallback.
- Anything about training throughput. The harness measures a single
  attention block, not an end-to-end training step.
- "Production-grade" or similar adjectives. The code was written
  with heavy AI assistance in a single session; the README's Status
  section documents exactly what Hridam has reviewed and what remains
  AI-generated.
