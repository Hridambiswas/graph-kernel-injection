# Next session

State on 23 Oct 2026 (end of autonomous mailbox loop):

- **Loop ended by STOP** at 23 Oct 01:09 IST (contents of
  `docs/mailbox/STOP`). The next session should begin by reading
  that file.
- Prompts **001 (read-only audit)**, **002 (history rebuild)** and
  **003 (correctness fixes + fair baselines + GPU-ready)** are all
  done. Reports are under `docs/mailbox/outbox/`.
- Prompt **004** landed in the inbox but was **not executed** —
  STOP took precedence per the loop protocol. 004 extends 003 with
  (a) an independent oracle rebuilt from the StreamingLLM definition
  directly, (b) M-1 ceiling-division alignment, (c) M-2 NaN guards
  for fully-masked rows, (d) M-3 explicit fp64/fp32 reporting, and
  (e) L-3 lazy-device + genuine FX idempotency. Pick it up next.
- Code has still **never run on GPU**. All numeric claims in
  `README.md` are analytical. The three-stage GPU run is wrapped
  by `scripts/gpu_run.sh`; the exact shapes and win/tie/loss
  criteria are frozen in `docs/eval_plan.md`.
- CPU tests: 28 cases under `tests/` pass on Mac without Triton
  (`python -m pytest tests/`).
- Branches: `main` has 1 honest root commit (`2562741`); work
  happens on `feat/gpu-verification`; the staged pre-rebuild chain
  is preserved local-only under `archive/staged-history`.
  **The archive branch must never be pushed.** Delete it with
  `git branch -D archive/staged-history` once Hridam is comfortable
  with the new history.
- No remote exists; nothing is pushed.
