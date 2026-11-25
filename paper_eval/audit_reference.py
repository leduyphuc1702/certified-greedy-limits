"""Check that FP64 argmax is the exact-real argmax: the FP64 top-1/top-2 gap must
exceed the sum of both rows' FP64 error bounds gamma_d(2^-53) * sum|w_k h_k|."""
import json
import sys

import numpy as np

sys.path.insert(0, "../v21_head_mips")
import killtests as kt  # noqa: E402

out = {}
for m in ("qwen", "tinyllama"):
    z = np.load(f"{m}.npz")
    W = kt.bf16_to_f64(z["W_bits"])
    H = z["h"].astype(np.float64)
    d = W.shape[1]
    g = d * 2.0**-53 / (1 - d * 2.0**-53)
    absW = np.abs(W)
    ok, min_ratio = 0, np.inf
    for lo in range(0, len(H), 64):
        S = H[lo:lo + 64] @ W.T
        E = g * (np.abs(H[lo:lo + 64]) @ absW.T)
        top = np.argsort(-S, axis=1)[:, :2]
        r = np.arange(len(S))
        gap = S[r, top[:, 0]] - S[r, top[:, 1]]
        # any rival j beats t only if S_t - S_j <= E_t + E_j; check all rivals
        Et = E[r, top[:, 0]][:, None]
        margin = S[r, top[:, 0]][:, None] - S - Et - E
        margin[r, top[:, 0]] = np.inf
        ok += int((margin.min(1) > 0).sum())
        min_ratio = min(min_ratio, float((gap / (2 * E.max(1))).min()))
    out[m] = {"positions": len(H), "reference_exact": ok, "min_gap_over_bound": min_ratio}
    print(m, out[m], flush=True)
json.dump(out, open("audit_reference.json", "w"), indent=1)
