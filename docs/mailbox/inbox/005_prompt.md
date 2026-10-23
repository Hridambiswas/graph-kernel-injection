# 005: Publish the repo on GitHub (public)

Hridam explicitly asked for the repo to be made public at this stopping point. That is
his approval to merge feat/gpu-verification into main through a pull request.

1. Preflight, stop and report if any fails:
   - 004 report exists and all CPU tests pass.
   - `git grep -nIiE "api[_-]?key|token|secret|password"` shows nothing sensitive; no
     .env files tracked.
   - `gh auth status` is logged in as Hridambiswas.
2. Delete the local backup of the staged history so it can never be pushed:
   `git branch -D archive/staged-history`. Mention it in the report.
3. Create the public repo: `gh repo create Hridambiswas/graph-kernel-injection --public
   --source=. --description "StreamingLLM block sparse attention in Triton (fwd+bwd)
   with a PyTorch FX injection pass. Correctness tests on CPU; GPU benchmarks pending."`
   If the name is taken, stop and report.
4. Push ONLY these two branches with explicit refspecs:
   `git push -u origin main` then `git push -u origin feat/gpu-verification`.
   Never use --all or --mirror.
5. Open a PR from feat/gpu-verification into main with a short honest description
   (what works, what is unverified, AI assisted), then merge it with a merge commit
   (`gh pr merge --merge`). Keep the feature branch.
6. Add topics: triton, pytorch, attention, flash-attention, compilers, torch-fx.
7. Report the repo URL, the PR URL, `git log --oneline -5 main`, and confirm
   `git branch -r` shows only origin/main and origin/feat/gpu-verification.

Stop. Write docs/mailbox/outbox/005_report.md, commit and push to feat/gpu-verification.
Then end the loop: there are no more prompts today.
