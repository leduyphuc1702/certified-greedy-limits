# v21 kill-tests on the LM head (preregistered 2026-11-25, thresholds from ChatGPT reply ../gpt_reply_02.md, before any run)

Shared: Qwen2.5-0.5B-Instruct tied head W (151,936 × 896, BF16 bits, exact in FP64). Queries = the 256 v20 test hidden states (final RMSNorm). Reference = exact-real greedy argmax of W·h in FP64. Pruning only when the upper bound is below the incumbent by more than a conservative FP64 guard.

Byte ratio ρ(h) = (weight bytes + metadata bytes read) / (2·V·d). Logical bytes, not measured DRAM traffic.

## v21a Exact MIPS tree (anisotropic block ball)
Binary tree, median split on the highest-variance coordinate, leaf ≤ 64 rows. Every node stores FP32 centroid c, 8 block radii R_g over contiguous 112-coordinate blocks plus one full radius. U(h) = min(c·h + Σ_g R_g‖h_g‖, c·h + R‖h‖). Best-first by U; leaves read exactly; incumbent from traversal only.
Metadata cost per visited node: 4·d + 4·9 + 8 bytes; per leaf row 4 bytes index.
PASS: mean ρ ≤ 15 %, P95 ρ ≤ 30 %, 0/256 wrong tokens, 256/256 certified. FAIL ⇒ drop this cluster-bound design.

## v21b Dimension-adaptive LM head
Coordinate order fixed from an independent calibration set (WikiText-2 validation + GSM8K train, 256 positions): descending E_cal[h_k²]·Σ_i W_ik². Checkpoints P = 64, 128, 256, 512, 896. Bound w_P·h_P ± ‖w_R‖‖h_R‖ with FP32 tail norms stored per row per checkpoint (4 × 4 bytes/row, read only for rows alive at that stage). After each stage the row with largest partial score is completed exactly to raise the incumbent.
PASS: mean ρ ≤ 25 %, P95 ρ ≤ 40 %, 0 wrong tokens, runtime ≤ 60 min.
