# Paper evaluation (preregistered 2026-11-25, before any run on these data)

Decision: one paper = preregistered negative results (v20–v22) + minimal remedy (v23). Thresholds below come from ChatGPT gpt_reply_02/03 and are frozen.

Models: Qwen/Qwen2.5-0.5B-Instruct (tied head, V=151,936, d=896) and TinyLlama/TinyLlama-1.1B-Chat-v1.0 (V=32,000, d=2,048).
Held-out positions per model: 1,024 = 128 prompts × 8 greedy tokens. Prompts: filtered WikiText-2 test paragraphs #16–#79 and GSM8K test questions #16–#79 (dev used #0–#15). Calibration: WikiText-2 validation #0–#15 + GSM8K train #0–#15. 95 % CIs by prompt-level bootstrap (2,000 resamples).

Tests and PASS thresholds (per model):
- v20 oracle box, drop 4 mantissa bits, both truncation and round-to-nearest: cert rate ≥ 95 %, 0 false certificates.
- v21a exact MIPS tree: mean ρ ≤ 15 %, P95 ≤ 30 %, 0 wrong.
- v21b dimension-adaptive, calibration order: mean ρ ≤ 25 %, P95 ≤ 40 %, 0 wrong.
- v21c FEXIPRO-style baseline: same as v21b but in the SVD basis of W (rows rotated by Vᵀ, singular-value order), bytes counted at 2 per coordinate (generous to the baseline). Same thresholds as v21b.
- v22 certified BF16-out head: refine ≤ 5 %, 0 wrong, 0 error-model violations.
- v23 certified FP32-out head, numerical part: e_i = γ_d·A_i (cheap A_i), refine ≤ 1 % at batch 1, 0 wrong, 0 violations. Timing part (overhead ≤ 10 % vs native BF16 head, ≤ 5 % vs uncertified FP32-out) requires a fused kernel and is evaluated later; until then v23 can only be "numerically PASS".
