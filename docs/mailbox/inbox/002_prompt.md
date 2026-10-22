# 002: Replace the staged history with an honest one

Hridam has approved this. The 68 commit history on main is staged (even 12 minute
spacing, messages describing a development arc that did not happen). It must not be
what anyone sees. Nothing is pushed, so this is a local change. Keep a backup so it
is reversible.

1. Confirm there is no remote (`git remote -v` empty) and nothing pushed. If a remote
   exists, STOP and write that as the report.
2. Backup: `git branch archive/staged-history main`. This branch must never be pushed.
   Add a note to docs/next_session.md saying it exists, is local only, and can be
   deleted once Hridam is happy.
3. Build a new history with real timestamps (no date overrides of any kind):
   - `git checkout --orphan new-main`, keep the current working tree.
   - One commit: "Initial implementation: StreamingLLM block sparse Triton attention
     (fwd+bwd), FX injection pass, benchmark harness. Written with AI assistance
     (Claude Code), not yet run on GPU." End with the Co-Authored-By line.
   - Rename: `git branch -M new-main main`.
4. Create the working branch `feat/gpu-verification` from the new main and switch to
   it. All further work happens there.
5. Show `git log --all --format='%h %ad %an %s' --date=iso` and `git branch -a` in the
   report.

Stop. Report in docs/mailbox/outbox/002_report.md, commit the report on
feat/gpu-verification.
