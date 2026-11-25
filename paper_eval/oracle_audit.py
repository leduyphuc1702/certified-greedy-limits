"""Rigorous classification of v20 oracle positions (reviewer P0).

For each position: proven accept (min_j B_j > 0 after FP64 error bound),
proven reject (some competitor has B_j <= 0 after error bound), or unresolved.
|w_t - w_j| and |h - h_hat| are exact in FP64 (BF16/FP32-representable inputs);
only the dot products carry rounding, bounded by g * (sum of absolute terms).
"""
import json
import sys

import numpy as np

sys.path.insert(0, "../v21_head_mips")
import killtests as kt  # noqa: E402

CHUNK = 8192


def audit(W, Wabs, H, Hc, ref):
    V, d = W.shape
    u = 2.0**-53
    g = 2 * (d + 4) * u / (1 - (d + 4) * u)
    acc = rej = unres = agree = 0
    for h, hh, r in zip(H, Hc, ref):
        rad = np.abs(h - hh)
        lc = W @ hh
        S = Wabs @ np.abs(hh)
        t = int(lc.argmax())
        agree += t == r
        lo_min, any_rej = np.inf, False
        for lo in range(0, V, CHUNK):
            A = np.abs(W[t] - W[lo:lo + CHUNK])
            m = lc[t] - lc[lo:lo + CHUNK]
            slack = A @ rad
            B = m - slack
            err = g * (S[t] + S[lo:lo + CHUNK] + slack + np.abs(m))
            o = np.arange(lo, lo + len(A)) != t
            lo_min = min(lo_min, (B - err)[o].min())
            any_rej |= bool(((B + err)[o] <= 0).any())
        if lo_min > 0:
            acc += 1
        elif any_rej:
            rej += 1
        else:
            unres += 1
    n = len(H)
    return {"proven_accept": acc, "proven_reject": rej, "unresolved": unres, "n": n,
            "ceiling_upper": (acc + unres) / n, "coarse_argmax_agree": agree / n}


def main():
    tag = sys.argv[1]
    z = np.load(f"{tag}.npz")
    W = kt.bf16_to_f64(z["W_bits"])
    Wabs = np.abs(W)
    H = z["h"].astype(np.float64)
    ref = (H @ W.T).argmax(1)
    out = {}
    for key in ("h_trunc", "h_rtn"):
        out[key] = audit(W, Wabs, H, z[key].astype(np.float64), ref)
        print(tag, key, json.dumps(out[key]), flush=True)
    json.dump(out, open(f"{tag}_oracle_audit.json", "w"), indent=1)


if __name__ == "__main__":
    main()
