"""Post-hoc (not preregistered): how well does a top-2 logit-margin trigger detect greedy flips
caused by low-precision weights (4 mantissa bits dropped in every projection)?

Flip at a position: argmax W h_coarse != argmax W h (exact argmax over the vocabulary, reference
hidden state h). The trigger sees only the coarse run: it fires when the top-1 minus top-2 logit
margin of W h_coarse is below tau. For each model and quantizer we report the fraction of positions
the trigger must fire at to catch every flip / 90% of flips, next to the oracle box certificate.
"""
import json
import sys

import numpy as np


def bf16(bits):
    return (bits.astype(np.int32) << 16).view(np.float32)


def top2_margin(W32, W64, H):
    out_arg, out_margin = [], []
    for lo in range(0, len(H), 64):
        L = H[lo:lo + 64].astype(np.float32) @ W32.T
        cand = np.argpartition(-L, 5, axis=1)[:, :5]
        for h, c in zip(H[lo:lo + 64], cand):
            exact = W64[c] @ h.astype(np.float64)
            o = np.argsort(-exact)
            out_arg.append(int(c[o[0]]))
            out_margin.append(float(exact[o[0]] - exact[o[1]]))
    return np.array(out_arg), np.array(out_margin)


res = {}
for tag in sys.argv[1:] or ["qwen", "tinyllama"]:
    z = np.load(f"{tag}.npz")
    W32 = bf16(z["W_bits"])
    W64 = W32.astype(np.float64)
    ref, ref_margin = top2_margin(W32, W64, z["h"])
    res[tag] = {}
    for q in ("trunc", "rtn"):
        arg, margin = top2_margin(W32, W64, z[f"h_{q}"])
        flip = arg != ref
        fm = np.sort(margin[flip])
        n = len(margin)

        def fire_rate(tau):
            return float((margin < tau).mean())

        tau_all = float(fm.max()) * (1 + 1e-12) if flip.any() else 0.0
        tau_90 = float(np.quantile(fm, 0.9)) * (1 + 1e-12) if flip.any() else 0.0
        r = dict(positions=n, flips=int(flip.sum()), flip_rate=float(flip.mean()),
                 flip_margin_median=float(np.median(fm)) if flip.any() else None,
                 flip_margin_max=float(fm.max()) if flip.any() else None,
                 all_margin_median=float(np.median(margin)),
                 fire_to_catch_all=fire_rate(tau_all), fire_to_catch_90=fire_rate(tau_90))
        res[tag][q] = r
        print(tag, q, json.dumps(r), flush=True)
json.dump(res, open("margin_detect.json", "w"), indent=1)
