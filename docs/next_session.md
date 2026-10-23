# Next session

State on 23 Oct 2026 (end of prompt 004).

## Status

- Prompts **001**, **002**, **003** and **004** are all **done**.
  Reports live under `docs/mailbox/outbox/`.
- Prompt **005** (publish the repo publicly on GitHub) is pending in
  the inbox. **005 is pre-approved by Hridam** per the prompt text;
  the mailbox loop may execute it directly on resume.
- Code has still **never run on GPU**. All numeric claims in
  `README.md` are analytical. The three-stage GPU run is wrapped by
  `scripts/gpu_run.sh`; the exact shapes and win/tie/loss criteria
  are frozen in `docs/eval_plan.md`.
- **The GPU run is the next real milestone** once 005 is done.
  Exact command on the GPU host:

  ```
  bash scripts/gpu_run.sh          # full-size shapes
  bash scripts/gpu_run.sh --small  # trimmed for an 8 GiB laptop GPU
  ```

  Results land under `results/<UTC-timestamp>/` and must not be
  edited after they are written.

## Verification state

- CPU tests: **58 cases pass, 1 GPU-only test skipped** on this Mac
  (`python -m pytest tests/`).
- Prompt 004 landed two critical correctness fixes that the earlier
  tests missed because the oracle and the token-mask helper shared
  the same bug:
  - **Bug A**: the mask formula was `sink | (causal & window)`,
    which let a query at position `q < sink_size` attend to future
    sink keys. Correct formula: `causal & (sink | window)`.
  - **Bug B**: `q_pos` started at 0 regardless of the KV cache
    length. For decode (`N_Q=1, N_KV=131072`) the lone query must
    sit at `N_KV - 1`, not 0, or it fakes a huge speedup by seeing
    only the sink prefix.
  Both are fixed in `sparse_triton_kernel.py` (kernel, oracle,
  `streaming_token_mask`, `make_streaming_mask`), in
  `benchmark.py` (FlexAttention `mask_mod`), and in the SDPA
  explicit-mask baseline (which routes through
  `streaming_token_mask`).
- A new hand-written reference (`tests/test_streaming_definition.py`)
  implements `visible(i, j) = j <= i and (j < S or i - j < W)` with
  `i = (N_KV - N_Q) + row` and checks every mask producer against
  it, including N_Q < S, non-block-divisible lengths (1000, 1023,
  1025), and decode `N_Q=1, N_KV=1000` (visible count must be
  `S + W` when they don't overlap).

## Branches

- `main`: 1 honest root commit (`2562741`), untouched since 002.
- Work branch: `feat/gpu-verification`.
- The staged pre-rebuild chain is preserved local-only under
  `archive/staged-history`. **The archive branch must never be
  pushed.** Prompt 005 asks you to delete it (`git branch -D
  archive/staged-history`) before publishing the repo.
- No remote yet; nothing is pushed. 005 creates the public GitHub
  remote.
