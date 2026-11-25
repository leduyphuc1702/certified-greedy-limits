# v22 kill-test: certified BF16 LM head decision (preregistered 2026-11-25, thresholds from ChatGPT ../gpt_reply_02.md "Hạng 2", before any run)

Question: can the native BF16 LM head (MPS, Qwen2.5-0.5B-Instruct) be made to return the exact-real argmax of W·h on every step, with a deterministic error bound and refinement only when needed, at low overhead?

- Queries: the 256 v20 test hidden states (BF16 values, final RMSNorm). Reference: exact-real argmax of W·h (FP64 is exact here: products of two BF16 fit in FP32, sum of 896 terms checked with FP64 accumulation error ≪ smallest observed gap; ties broken by lowest index).
- Native path: torch.matmul on MPS, BF16 in, BF16 out (what the model's lm_head does).
- Error model (an assumption about the kernel, checked empirically below): FP32 accumulation in any order, then one round-to-nearest to BF16.
  |ŝ_i − s_i| ≤ e_i = γ_d·A_i·(1+u) + u·|ŝ_i|/(1−u), u = 2^-8 (BF16), γ_d = d·2^-24/(1 − d·2^-24), A_i ≥ Σ_k |w_ik h_k|.
  Cheap A_i = min(‖w_i‖₂‖h‖₂, ‖w_i‖_∞‖h‖₁) with precomputed row norms (O(V) extra, no second matvec).
- Certificate: t = argmax ŝ. Candidate set C = {i : ŝ_i + e_i ≥ ŝ_t − e_t}. If C = {t}: certified, return t. Else refine C exactly (FP64 on CPU) and return its argmax.
- Model check: count rows where |ŝ_i − s_i| > e_i (any such row ⇒ the kernel error model is wrong ⇒ FAIL regardless of the rest).

PASS requires all of:
1. wrong tokens after correction vs exact-real: 0/256
2. queries needing refinement ≤ 5 %
3. overhead of (native head + certificate check, refinement excluded) vs native head alone ≤ 10 %, median of timed repetitions on MPS
4. zero error-model violations
Runtime ≤ 90 min.

Diagnostics: mean |C|, native-vs-exact mismatch count (expected ≈ 2 % from v20), refinement rate with the exact A_i = |W|·|h| instead of the cheap bound.
