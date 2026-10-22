# 003: Make it honest and GPU ready (on feat/gpu-verification)

Read 001_report.md and docs/architecture_notes.md first. Confirm you are on
feat/gpu-verification. One commit per logical change, real timestamps.

1. Fix the HIGH risk correctness issues from 001, one at a time. For each: explain the
   cause in architecture_notes.md, make one targeted fix, and add a CPU runnable test
   where possible (for example a test that the token level mask used by the oracle
   equals the mask the kernel implies, including causal masking inside diagonal
   blocks). If a fix can only be confirmed on GPU, mark it "fix written, unconfirmed
   until GPU run".

2. Fair baselines in benchmark.py, with the SAME token level mask:
   - torch.nn.attention.flex_attention with a block mask for sinks + sliding window.
   - F.scaled_dot_product_attention with an explicit boolean mask, and plain causal
     SDPA (dense FlashAttention backend) as the "no sparsity" reference.
   - Dense eager reference.
   Report fwd and fwd+bwd latency (p50, p95), peak memory, and max abs error vs the
   fp64 reference for every method. Bump requirements to torch>=2.5 if FlexAttention
   needs it.

3. Widen the FX pass so it also matches F.scaled_dot_product_attention calls (with
   is_causal=True), not only the repo's own hand written pattern. Add a test that
   traces a small module using SDPA and checks the node gets replaced. Do not claim
   Hugging Face support unless a test proves it.

4. scripts/gpu_run.sh: one command for tomorrow. It must
   - print nvidia-smi, python, torch, triton versions and GPU name into
     results/<timestamp>/env.txt,
   - run verify-grads, then verify-fwd, then the benchmark, teeing raw output into
     results/<timestamp>/,
   - stop at the first failure and keep the log,
   - have a --small flag with shapes that fit an 8 GB laptop GPU.

5. Before any GPU run, write docs/eval_plan.md: the exact shapes, sink and window
   sizes, dtypes, number of warmup and timed iterations, and what counts as a win,
   tie or loss vs FlexAttention. Commit it before the run so results can't shape
   the plan.

6. README: add a short "Status" section at the top: what is verified, what is not,
   "Written with heavy AI assistance (Claude Code); Hridam reviewed X and verified Y"
   (leave X and Y as TODO for Hridam to fill honestly). Replace every unmeasured
   number or adjective (massive, elite, production grade and so on) with neutral
   wording or "not yet measured". Remove the "benchmark results template" if it
   contains invented numbers.

Stop. Report in docs/mailbox/outbox/003_report.md, commit on feat/gpu-verification.
