# v20 kill-test: Oracle Box Certificate (preregistered 2026-11-25, before any run)

Designed by ChatGPT (chat "Tiếp tục v19", reply saved in ../gpt_reply_01.md), run by Claude on Apple M4.

- Model: Qwen/Qwen2.5-0.5B-Instruct, reference = original BF16 weights.
- Coarse trunk: drop the 4 low fraction bits (truncate toward zero) of every q/k/v/o/gate/up/down weight. Embedding, RMSNorm, LM head unchanged.
- Data: first 16 WikiText-2 test paragraphs (>= 40 words, not headings), first 16 GSM8K test questions (chat template). Prompts cut to 48 tokens.
- Positions: 8 greedy tokens per prompt from the reference model, 256 positions. Both models see the same reference prefix.
- h, ĥ = final-RMSNorm hidden state (reference, coarse). Oracle radius r_k = |h_k − ĥ_k|.
- t = argmax_j w_j·ĥ. B_j = (w_t − w_j)·ĥ − Σ_k |w_tk − w_jk| r_k over the whole vocabulary, in FP64 with a conservative rounding guard. Certified iff min_{j≠t} B_j > 0.

PASS requires all of:
1. oracle certification rate >= 95 %
2. zero false certificates (certified t ≠ exact-real argmax of W·h)
3. zero certified tokens that differ from the native BF16 token
4. all 256 positions completed

FAIL on certification rate ⇒ drop whole-Transformer certification by symmetric coordinate-wise error propagation (including tighter bounds in the same representation).

Diagnostics (not part of the verdict): drop 1/2/3 bits, margin distribution, headroom s* = min_j margin_j / slack_j (how much the oracle radius could be inflated before the certificate breaks).
