"""Stage 2: all preregistered tests on one model's held-out dump."""
import json
import sys
import time

import numpy as np
import torch

sys.path.insert(0, "../v21_head_mips")
import killtests as kt  # noqa: E402

GEN = 8
U = 2.0**-8
EPS64 = np.finfo(np.float64).eps
CHUNK = 8192


def boot(per_query, reps=2000, seed=0):
    x = np.asarray(per_query, dtype=np.float64).reshape(-1, GEN).mean(1)
    rng = np.random.default_rng(seed)
    m = x[rng.integers(0, len(x), (reps, len(x)))].mean(1)
    return [float(x.mean()), float(np.quantile(m, 0.025)), float(np.quantile(m, 0.975))]


def oracle(W, H, Hc, ref):
    cert, false, head = [], 0, []
    V, d = W.shape
    for h, hh, r in zip(H, Hc, ref):
        rad = np.abs(h - hh)
        lc = W @ hh
        t = int(lc.argmax())
        bmin, hmin = np.inf, np.inf
        for lo in range(0, V, CHUNK):
            A = np.abs(W[t] - W[lo:lo + CHUNK])
            margin = lc[t] - lc[lo:lo + CHUNK]
            slack = A @ rad
            B = margin - slack - 4 * d * EPS64 * (A @ np.abs(hh) + slack)
            o = np.arange(lo, lo + len(A)) != t
            bmin = min(bmin, B[o].min())
            with np.errstate(divide="ignore", invalid="ignore"):
                hmin = min(hmin, np.where(slack[o] > 0, margin[o] / slack[o], np.inf).min())
        ok = bool(bmin > 0)
        cert.append(ok)
        false += ok and t != r
        head.append(float(hmin))
    return {"cert": boot(cert), "false": int(false), "headroom_median": float(np.median(head))}


def checkpoints(d):
    cps, p = [], 64
    while p < d:
        cps.append(p)
        p *= 2
    return tuple(cps + [d])


def dim_adaptive(Wp, H, ref_fn, d):
    kt.CHECKPOINTS = checkpoints(d)
    tails = np.stack([np.linalg.norm(Wp[:, P:], axis=1) for P in kt.CHECKPOINTS[:-1]], 1)
    tails = np.nextafter(tails.astype(np.float32), np.float32(np.inf)).astype(np.float64)
    rho, wrong = [], 0
    for q, h in enumerate(H):
        j, b = kt.dim_adaptive(Wp, tails, h, d)
        rho.append(b / (2 * len(Wp) * d))
        wrong += not ref_fn(q, j)
    return {"mean_rho": boot(rho), "p95_rho": float(np.quantile(rho, 0.95)), "wrong": int(wrong)}


def tree(W, H, ref):
    d = W.shape[1]
    blocks = [np.arange(g * d // 8, (g + 1) * d // 8) for g in range(8)]
    root = kt.build(W, np.arange(len(W)), blocks)
    rho, rows, wrong = [], [], 0
    for h, r in zip(H, ref):
        j, b, nr = kt.tree_query(root, W, h, blocks, d)
        rho.append(b / (2 * len(W) * d))
        rows.append(nr / len(W))
        wrong += j != r
    return {"mean_rho": boot(rho), "p95_rho": float(np.quantile(rho, 0.95)), "rows": float(np.mean(rows)), "wrong": int(wrong)}


def head_cert(Wbits, W, H, exact, ref, fp32_out):
    d = W.shape[1]
    gamma = d * 2.0**-24 / (1 - d * 2.0**-24)
    Wt = torch.from_numpy(Wbits).view(torch.bfloat16).to("mps")
    if fp32_out:
        Wt = Wt.float()
    n2 = np.linalg.norm(W, axis=1) * (1 + 1e-6)
    ninf = np.abs(W).max(1)
    refine, wrong, viol, ncand = [], 0, 0, []
    for q, h in enumerate(H):
        ht = torch.from_numpy(h.astype(np.float32)).to("mps")
        s = (ht @ Wt.T) if fp32_out else (ht.to(torch.bfloat16) @ Wt.T)
        s = s.float().cpu().double().numpy()
        A = np.minimum(n2 * np.linalg.norm(h), ninf * np.abs(h).sum())
        e = gamma * A * (1 + 1e-6)
        if not fp32_out:
            e = e * (1 + U) + U * np.abs(s) / (1 - U)
        viol += int((np.abs(s - exact[q]) > e).sum())
        t = int(s.argmax())
        cand = np.nonzero(s + e >= s[t] - e[t])[0]
        ncand.append(len(cand))
        refine.append(len(cand) > 1)
        out = int(cand[(W[cand] @ h).argmax()]) if len(cand) > 1 else t
        wrong += out != ref[q]
    return {"refine": boot(refine), "wrong": int(wrong), "violations": viol, "mean_cand": float(np.mean(ncand))}


def main():
    tag = sys.argv[1]
    z = np.load(f"{tag}.npz")
    W = kt.bf16_to_f64(z["W_bits"])
    H = z["h"].astype(np.float64)
    V, d = W.shape
    exact = H @ W.T
    ref = exact.argmax(1)
    res = {"V": V, "d": d, "positions": len(H), "native_vs_exact_mismatch": boot(z["native"] != ref)}
    steps = [
        ("v20_trunc", lambda: oracle(W, H, z["h_trunc"].astype(np.float64), ref)),
        ("v20_rtn", lambda: oracle(W, H, z["h_rtn"].astype(np.float64), ref)),
        ("v22_bf16_out", lambda: head_cert(z["W_bits"], W, H, exact, ref, False)),
        ("v23_fp32_out", lambda: head_cert(z["W_bits"], W, H, exact, ref, True)),
    ]
    Hc = z["h_cal"].astype(np.float64)
    perm = np.argsort(-((Hc**2).mean(0) * (W**2).sum(0)), kind="stable")
    steps.append(("v21b_dim_adaptive", lambda: dim_adaptive(W[:, perm], H[:, perm], lambda q, j: j == ref[q], d)))

    def svd_basis():
        evals, evecs = np.linalg.eigh(W.T @ W)
        Vt = evecs[:, np.argsort(-evals)].T
        return dim_adaptive(W @ Vt.T, H @ Vt.T, lambda q, j: j == ref[q], d)

    steps.append(("v21c_svd_fexipro", svd_basis))
    steps.append(("v21a_tree", lambda: tree(W, H, ref)))
    only = set(sys.argv[2:])
    for name, fn in steps:
        if only and name not in only:
            continue
        t0 = time.time()
        res[name] = fn() | {"seconds": round(time.time() - t0, 1)}
        print(tag, name, json.dumps(res[name]), flush=True)
        json.dump(res, open(f"{tag}_results.json", "w"), indent=1)


if __name__ == "__main__":
    main()
