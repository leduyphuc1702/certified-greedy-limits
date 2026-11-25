import json
import time

import numpy as np
import torch

U = 2.0**-8
DEV = "mps"


def bf16_to_f64(bits):
    return (bits.astype(np.int32).astype(np.uint32) << 16).view(np.float32).astype(np.float64)


def sync():
    torch.mps.synchronize()


def main():
    z = np.load("../v21_head_mips/hidden.npz")
    W64 = bf16_to_f64(z["W_bf16_bits"])
    H64 = z["h_test"].astype(np.float64)
    V, d = W64.shape
    gamma = d * 2.0**-24 / (1 - d * 2.0**-24)

    exact = H64 @ W64.T
    ref = exact.argmax(1)

    Wt = torch.from_numpy(z["W_bf16_bits"]).view(torch.bfloat16).to(DEV)
    Ht = torch.from_numpy(H64.astype(np.float32)).to(torch.bfloat16).to(DEV)
    assert np.array_equal(Ht.float().cpu().double().numpy(), H64)

    n2 = torch.from_numpy(np.linalg.norm(W64, axis=1).astype(np.float32) * (1 + 1e-6)).to(DEV)
    ninf = torch.from_numpy(np.abs(W64).max(1).astype(np.float32)).to(DEV)
    Wabs = np.abs(W64)

    def native(h):
        return h @ Wt.T

    def check(h, s):
        s32 = s.float()
        A = torch.minimum(n2 * torch.linalg.vector_norm(h.float()), ninf * h.float().abs().sum())
        e = gamma * A * (1 + 1e-6) * (1 + U) + U * s32.abs() / (1 - U)
        t = torch.argmax(s32)
        floor = s32[t] - e[t]
        ncand = int(((s32 + e) >= floor).sum())
        return int(t), ncand, e

    rows = []
    violations = 0
    for q in range(len(H64)):
        s = native(Ht[q])
        t, ncand, e = check(Ht[q], s)
        s64 = s.float().cpu().double().numpy()
        e64 = e.cpu().double().numpy()
        violations += int((np.abs(s64 - exact[q]) > e64).sum())
        if ncand == 1:
            out = t
        else:
            cand = np.nonzero(s64 + e64 >= s64[t] - e64[t])[0]
            out = int(cand[(W64[cand] @ H64[q]).argmax()])
        A_exact = Wabs @ np.abs(H64[q])
        e_ex = gamma * A_exact * (1 + U) + U * np.abs(s64) / (1 - U)
        ncand_exact = int((s64 + e_ex >= s64[t] - e_ex[t]).sum())
        rows.append({"native": t, "out": out, "ref": int(ref[q]), "ncand": ncand, "ncand_exact_A": ncand_exact})

    reps = 200
    for _ in range(20):
        native(Ht[0])
    sync()
    t_nat, t_cert = [], []
    for r in range(reps):
        h = Ht[r % len(Ht)]
        sync()
        a = time.perf_counter()
        s = native(h)
        torch.argmax(s)
        sync()
        t_nat.append(time.perf_counter() - a)
        sync()
        a = time.perf_counter()
        s = native(h)
        t, ncand, _ = check(h, s)
        sync()
        t_cert.append(time.perf_counter() - a)

    refine = np.mean([r["ncand"] > 1 for r in rows])
    summary = {
        "wrong_after_correction": int(sum(r["out"] != r["ref"] for r in rows)),
        "native_vs_exact_mismatch": int(sum(r["native"] != r["ref"] for r in rows)),
        "refinement_rate": float(refine),
        "refinement_rate_exact_A": float(np.mean([r["ncand_exact_A"] > 1 for r in rows])),
        "mean_candidates": float(np.mean([r["ncand"] for r in rows])),
        "error_model_violations": violations,
        "native_median_ms": 1e3 * float(np.median(t_nat)),
        "certified_median_ms": 1e3 * float(np.median(t_cert)),
        "overhead": float(np.median(t_cert) / np.median(t_nat) - 1),
    }
    print(json.dumps(summary, indent=1))
    json.dump({"summary": summary, "rows": rows}, open("results.json", "w"), indent=1)


if __name__ == "__main__":
    main()
