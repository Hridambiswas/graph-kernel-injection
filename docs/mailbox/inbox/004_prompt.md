# 004: Correctness fixes to a clean stopping point (no GPU work)

Change of plan from Hridam: GPU testing is postponed. Goal today is to leave the repo
correct as far as CPU can prove, honest about what is unverified, and published.
This replaces the earlier 004 draft. Work on feat/gpu-verification.

## 1. CRITICAL mask bugs found in director review of 003 (do first)

The CPU tests passed only because the reference `streaming_token_mask` shares these
bugs, the H-2 problem again.

A. Sinks leak the future. The mask is written as `sink | (causal & windowed)` in the
   kernel (~295, ~454, ~557), the oracle (~842), `streaming_token_mask` (~878) and the
   FlexAttention mask_mod. A query q < sink_size can see sink keys k with
   q < k < sink_size. Correct: `causal & (sink | windowed)`, so k <= q always. Fix every
   site and the docstring formula.

B. Query positions ignore the KV offset. q_pos starts at 0. For decode (N_Q = 1,
   N_KV = 131072) the query must sit at N_KV - 1: q_pos = (N_KV - N_Q) + arange(N_Q),
   the bottom right aligned convention FlashAttention and FlexAttention use with a KV
   cache. Today a decode query is treated as token 0 and attends to almost nothing,
   which would fake a huge speedup. Fix kernel (q_off), oracle, streaming_token_mask,
   block mask builder, FlexAttention mask_mod and the SDPA explicit mask together.

C. Independent tests written from the definition by hand, calling no repo helper:
   `visible(i, j) = j <= i and (j < S or i - j < W)` with i = (N_KV - N_Q) + row.
   Check every mask producer against it, including N_Q < S, N not divisible by the
   block size (1000, 1023, 1025), and decode N_Q = 1, N_KV = 1000 where the visible
   count must be S + W (when they don't overlap), not about S. Keep a test that the
   block mask is a superset of the token mask (no needed block skipped).

## 2. Cheap medium fixes
- M-1: mask builder and autograd bridge both use ceiling division.
- M-2: fully masked rows give zero output and zero grads, never NaN (guard the log and
  the backward exp). The test may be GPU only, skipped without CUDA.
- M-3: verify-grads prints "FP64 GRADCHECK NOT RUN: <reason>" when it falls back,
  records fp64_ran=false, and exits nonzero if any check that ran fails.
- L-3: build the mask lazily on the input's device so tracing works on CPU.

## 3. Housekeeping
- Add `.claude/settings.local.json` and `.claude/scheduled_tasks.lock` to .gitignore.
- README Status block: "Kernel and FX pass implemented; mask semantics verified by CPU
  tests; NOT yet run on GPU, no performance numbers yet; GPU verification planned."
  Keep the AI assistance line. Leave the Hridam TODO fields for him.
- docs/next_session.md: GPU run is the next milestone; list the exact command.

Run the full CPU test suite and paste raw pytest output in the report.

Stop. Report in docs/mailbox/outbox/004_report.md, commit on feat/gpu-verification.
