# 001: Read, map and audit. Change no code.

Context: you wrote this repo yesterday in one session. A new director (a separate
Claude chat) now reviews your work. Read CLAUDE.md first, especially the honesty rules,
which are new. This step changes NO code and makes no history changes.

1. Read every file fully: sparse_triton_kernel.py, fx_compiler_pass.py, benchmark.py,
   README.md, pyproject.toml, requirements.txt. Run `git log --format='%h %ad %s'
   --date=iso | head -80` and `git branch -a`.

2. Write docs/architecture_notes.md covering:
   - What each module does and how they connect (mask builders, fwd kernel, the three
     bwd kernels, autograd Function, FX injector, benchmark CLI modes).
   - Exactly which graph patterns the FX pass can match today. Be blunt: does it match
     only the repo's own StandardAttention (matmul, scale, softmax, matmul)? Does it
     match F.scaled_dot_product_attention, nn.MultiheadAttention, or attention in a
     Hugging Face model (for example GPT-2 or Llama)? If not, say so.
   - Every claim in README.md that has not been measured on hardware, listed with line
     numbers.

3. Correctness audit by reading code (you cannot run Triton on this Mac). For each,
   point to file and line and rate risk high, medium or low:
   - Bounds masks on loads and stores when N is not a multiple of block_m / block_n.
   - Fully masked rows (a query block whose kv blocks are all skipped): division by
     l_i, log of zero, NaN in lse, and what the bwd does with them.
   - Causality: StreamingLLM is causal. Is the causal mask applied inside the diagonal
     blocks, or only at block granularity? If only block granularity, tokens can see the
     future inside a block. That would be a correctness bug, not a tuning detail.
   - Whether the dense reference oracle uses exactly the same token level mask as the
     kernel, so gradcheck compares like with like.
   - Whether gradcheck runs Triton in fp64 for real, or silently falls back.
   - Anything else that would crash or give wrong numbers on first GPU contact.
   Do NOT fix anything yet. Diagnosis only.

4. List what is needed to run on an NVIDIA laptop tomorrow: OS assumptions (Triton is
   Linux only officially; on Windows it needs WSL2), minimum torch / triton versions,
   GPU memory needed for the default benchmark shapes, and which shapes to shrink for an
   8 GB laptop GPU.

Stop. Report in docs/mailbox/outbox/001_report.md. Commit only the two new docs files
(architecture_notes.md and the report) plus CLAUDE.md, docs/next_session.md,
.claude/settings.json and the mailbox folders, with a normal current timestamp, on main
for now (there is no remote, and 002 rebuilds history from scratch). Do not push.
