import heapq
import json
import time

import numpy as np

D_BLOCKS = 8
LEAF = 64
CHECKPOINTS = (64, 128, 256, 512, 896)
GUARD = 1e-9


def bf16_to_f64(bits):
    return (bits.astype(np.int32).astype(np.uint32) << 16).view(np.float32).astype(np.float64)


class Node:
    __slots__ = ("idx", "c", "R", "Rg", "left", "right")


def build(W, idx, blocks):
    n = Node()
    X = W[idx]
    n.idx = idx
    n.c = X.mean(0).astype(np.float32).astype(np.float64)
    E = X - n.c
    n.R = float(np.sqrt((E * E).sum(1)).max()) * (1 + 1e-6)
    n.Rg = np.array([np.sqrt((E[:, b] ** 2).sum(1)).max() for b in blocks]) * (1 + 1e-6)
    n.left = n.right = None
    if len(idx) > LEAF:
        k = int(X.var(0).argmax())
        order = np.argsort(X[:, k], kind="stable")
        half = len(idx) // 2
        n.left = build(W, idx[order[:half]], blocks)
        n.right = build(W, idx[order[half:]], blocks)
    return n


def tree_query(root, W, h, blocks, d):
    hn = np.linalg.norm(h)
    hg = np.array([np.linalg.norm(h[b]) for b in blocks])
    meta_node = 4 * d + 4 * 9 + 8

    def ub(n):
        base = n.c @ h
        u = min(base + n.Rg @ hg, base + n.R * hn)
        return u + GUARD * (np.abs(n.c) @ np.abs(h) + n.R * hn + 1)

    bytes_read, rows_read = meta_node, 0
    heap = [(-ub(root), 0, root)]
    cnt = 1
    best, best_i = -np.inf, -1
    while heap:
        negu, _, n = heapq.heappop(heap)
        if -negu < best:
            break
        if n.left is None:
            s = W[n.idx] @ h
            j = int(s.argmax())
            rows_read += len(n.idx)
            bytes_read += len(n.idx) * (2 * d + 4)
            if s[j] > best or (s[j] == best and n.idx[j] < best_i):
                best, best_i = float(s[j]), int(n.idx[j])
            continue
        for ch in (n.left, n.right):
            bytes_read += meta_node
            heapq.heappush(heap, (-ub(ch), cnt, ch))
            cnt += 1
    return best_i, bytes_read, rows_read


def dim_adaptive(Wp, tails, h, d):
    V = len(Wp)
    alive = np.arange(V)
    S = np.zeros(V)
    prev = 0
    bytes_read = 0
    L = -np.inf
    for ci, P in enumerate(CHECKPOINTS):
        S[alive] += Wp[alive, prev:P] @ h[prev:P]
        bytes_read += len(alive) * (P - prev) * 2
        if P == d:
            j = alive[S[alive].argmax()]
            return int(j), bytes_read
        bytes_read += len(alive) * 4
        hr = np.linalg.norm(h[P:])
        slack = tails[alive, ci] * hr
        g = GUARD * (np.abs(S[alive]) + slack + 1)
        top = alive[S[alive].argmax()]
        exact_top = S[top] + Wp[top, P:] @ h[P:]
        bytes_read += (d - P) * 2
        L = max(L, exact_top, float((S[alive] - slack - g).max()))
        alive = alive[S[alive] + slack + g >= L]
        prev = P


def summarize(rho, wrong, n):
    return {
        "mean_rho": float(np.mean(rho)),
        "p95_rho": float(np.quantile(rho, 0.95)),
        "median_rho": float(np.median(rho)),
        "wrong_tokens": int(wrong),
        "queries": n,
    }


def main():
    z = np.load("hidden.npz")
    W = bf16_to_f64(z["W_bf16_bits"])
    H = z["h_test"].astype(np.float64)
    Hc = z["h_cal"].astype(np.float64)
    V, d = W.shape
    ref = (H @ W.T).argmax(1)
    out = {}

    t0 = time.time()
    blocks = [np.arange(g * d // D_BLOCKS, (g + 1) * d // D_BLOCKS) for g in range(D_BLOCKS)]
    root = build(W, np.arange(V), blocks)
    rho, rows, wrong = [], [], 0
    for h, r in zip(H, ref):
        j, b, nr = tree_query(root, W, h, blocks, d)
        rho.append(b / (2 * V * d))
        rows.append(nr / V)
        wrong += j != r
    out["v21a_tree"] = summarize(rho, wrong, len(H)) | {"mean_rows_frac": float(np.mean(rows)), "seconds": time.time() - t0}
    print("v21a", json.dumps(out["v21a_tree"]), flush=True)

    t0 = time.time()
    energy = (Hc**2).mean(0) * (W**2).sum(0)
    perm = np.argsort(-energy, kind="stable")
    Wp = W[:, perm]
    tails = np.stack([np.linalg.norm(Wp[:, P:], axis=1) for P in CHECKPOINTS[:-1]], 1)
    tails = tails.astype(np.float32).astype(np.float64) * (1 + 1e-6)
    rho, wrong = [], 0
    for h, r in zip(H, ref):
        j, b = dim_adaptive(Wp, tails, h[perm], d)
        rho.append(b / (2 * V * d))
        wrong += j != r
    out["v21b_dim_adaptive"] = summarize(rho, wrong, len(H)) | {"seconds": time.time() - t0}
    print("v21b", json.dumps(out["v21b_dim_adaptive"]), flush=True)
    json.dump(out, open("killtest_results.json", "w"), indent=1)


if __name__ == "__main__":
    main()
