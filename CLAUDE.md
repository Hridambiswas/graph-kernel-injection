# graph-kernel-injection

StreamingLLM style block sparse attention (attention sinks + sliding window) as a
Triton kernel with forward and backward passes, plus a PyTorch FX pass that finds
attention in a model graph and swaps in the kernel. Purpose: an honest, verified
proof of work asset for Hridam's outreach to ML systems professors (UK, Germany,
Netherlands). It is only worth anything if every claim in it is true and checked.

Roles: a separate Claude chat is the director (thinks, reviews, writes prompts in
docs/mailbox/inbox/). You, Claude Code, write code and run things. Hridam is the
supervisor and makes the real decisions.

## HARD CONSTRAINTS (never break)

1. Understand before changing. Before any change, read the whole architecture and
   structure: README, docs, directory tree, entrypoints, configs, how data flows
   between components, and the recent git history. Write down (in
   docs/architecture_notes.md and in the report) what the system does, where the
   relevant logic lives, and exactly what needs to change and why. Only then change
   code. No hit and trial: no speculative edits, no "try this and see" loops. Every
   change must be justified by evidence (logs, reproduction, code reading).
2. Never push directly to main (or master). Check the existing branches first. Work
   on the proper sub branch for the task (for example feat/<topic>, fix/<topic>,
   research/<topic>). Push to that branch only. Merging into main happens only when
   Hridam explicitly asks, preferably through a pull request.

## Honesty rules for this repo (equally hard)

- Never fabricate, backdate or "simulate" git history. Commit timestamps are the
  real time of the commit. Never set GIT_AUTHOR_DATE or GIT_COMMITTER_DATE, never
  use --date, never rewrite history to look older or longer than it was.
- Commit messages describe what actually changed. One commit per logical change.
  Do not chase commit counts.
- Every commit you author ends with the Co-Authored-By line for Claude. The README
  states plainly that the code was written with heavy AI assistance and what Hridam
  verified himself.
- No performance or correctness claim goes in the README, a docstring or a commit
  message until a script in this repo produced it on real hardware and the raw
  output is saved under results/. Until then write "not yet measured".
- Baselines must be fair: compare against PyTorch FlexAttention and SDPA
  (FlashAttention / memory efficient backends) with the same mask, not only against
  dense eager attention. Report losses and ties as plainly as wins.
- Never edit files under results/ after they are written. Re-runs go in a new
  timestamped folder.
- Never commit secrets, tokens or .env files. Never print keys in reports.

## Every session

1. Read docs/next_session.md.
2. Confirm the working branch is not main (`git branch --show-current`).
3. Stop and report after each numbered step of a prompt if the prompt says so.
4. At the end, update docs/next_session.md.

## Mailbox protocol

- Prompts arrive as docs/mailbox/inbox/NNN_prompt.md. Execute the lowest NNN that has
  no matching docs/mailbox/outbox/NNN_report.md.
- Write the full report to docs/mailbox/outbox/NNN_report.md: what you did, exact
  commands, raw numbers, what failed, open questions. Commit and push to the working
  branch (never main). If there is no remote yet, just commit.
- Never edit inbox files. If docs/mailbox/STOP exists, end the loop.
- If a prompt is ambiguous or would break this file, write your question as the
  report instead of guessing.

## Autonomy

Every prompt in docs/mailbox/inbox/ is pre-approved, including long runs. If a
previous session left partial work for a prompt, commit the scripts, reuse valid
partial results and resume without waiting. Keep the loop going. Only stop for a
STOP file, a rule in this file, or a genuine ambiguity (write it as the report).
Autonomy never overrides the hard constraints or the honesty rules: no changes before
the architecture is understood, no pushes to main, no fabricated history or numbers.
