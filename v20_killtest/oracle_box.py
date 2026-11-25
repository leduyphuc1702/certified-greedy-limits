import json
import os
import sys
import time

import numpy as np
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
PROMPT_TOKENS = 48
GEN = 8
PROJ = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")
DEV = "mps" if torch.backends.mps.is_available() else "cpu"
EPS64 = np.finfo(np.float64).eps
ROUND = os.environ.get("GITI_ROUND") == "1"


def prompts(tok):
    wiki = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")
    out = []
    for row in wiki:
        s = row["text"].strip()
        if s and not s.startswith("=") and len(s.split()) >= 40:
            out.append(("wiki", tok(s)["input_ids"][:PROMPT_TOKENS]))
        if len(out) == 16:
            break
    gsm = load_dataset("openai/gsm8k", "main", split="test")
    for row in gsm.select(range(16)):
        ids = tok.apply_chat_template(
            [{"role": "user", "content": row["question"]}], add_generation_prompt=True
        )
        if not isinstance(ids, list):
            ids = ids["input_ids"]
        out.append(("gsm8k", ids[-PROMPT_TOKENS:]))
    return out


def load(drop_bits):
    m = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16).to(DEV).eval()
    if drop_bits:
        mask = ~((1 << drop_bits) - 1)
        with torch.no_grad():
            for name, mod in m.named_modules():
                if name.split(".")[-1] in PROJ:
                    w = mod.weight.data
                    b = w.view(torch.int16)
                    if ROUND:
                        b = b + (1 << (drop_bits - 1))
                    w.copy_((b & mask).view(torch.bfloat16))
    return m


@torch.no_grad()
def final_hidden(m, seqs):
    hs, logits = [], []
    for ids, p in seqs:
        x = torch.tensor([ids], device=DEV)
        o = m(x, output_hidden_states=True)
        sl = slice(p - 1, p - 1 + GEN)
        hs.append(o.hidden_states[-1][0, sl].float().cpu().double().numpy())
        logits.append(o.logits[0, sl].float().argmax(-1).cpu().numpy())
    return np.concatenate(hs), np.concatenate(logits)


def certify(W, h, hh):
    r = np.abs(h - hh)
    lc = W @ hh
    t = int(lc.argmax())
    D = W[t] - W
    A = np.abs(D)
    margin = D @ hh
    slack = A @ r
    guard = 4 * W.shape[1] * EPS64 * (A @ np.abs(hh) + slack)
    B = margin - slack - guard
    B[t] = np.inf
    others = np.arange(len(W)) != t
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(slack[others] > 0, margin[others] / slack[others], np.inf)
    lc_sorted = np.sort(lc)
    return {
        "t": t,
        "certified": bool(B.min() > 0),
        "headroom": float(ratio.min()),
        "margin12": float(lc_sorted[-1] - lc_sorted[-2]),
        "exact_argmax": int((W @ h).argmax()),
        "err_inf": float(r.max()),
        "err_l2": float(np.linalg.norm(r)),
    }


def main():
    drops = [int(a) for a in sys.argv[1:]] or [4]
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(MODEL)
    ref = load(0)
    seqs = []
    for src, ids in prompts(tok):
        x = torch.tensor([ids], device=DEV)
        g = ref.generate(x, attention_mask=torch.ones_like(x), max_new_tokens=GEN, min_new_tokens=GEN, do_sample=False)
        seqs.append((g[0].tolist(), len(ids)))
    h_ref, native = final_hidden(ref, seqs)
    W = ref.lm_head.weight.detach().float().cpu().double().numpy()
    del ref
    report = {"model": MODEL, "device": DEV, "positions": len(h_ref), "runs": {}}
    for k in drops:
        m = load(k)
        h_c, native_c = final_hidden(m, seqs)
        del m
        rows = [certify(W, h_ref[i], h_c[i]) for i in range(len(h_ref))]
        cert = [r["certified"] for r in rows]
        head = np.array([r["headroom"] for r in rows])
        summary = {
            "certification_rate": float(np.mean(cert)),
            "false_certificates": int(sum(r["certified"] and r["t"] != r["exact_argmax"] for r in rows)),
            "certified_vs_native_mismatch": int(sum(r["certified"] and r["t"] != native[i] for i, r in enumerate(rows))),
            "coarse_argmax_agreement_exact": float(np.mean([r["t"] == r["exact_argmax"] for r in rows])),
            "coarse_native_agreement": float(np.mean(native_c == native)),
            "exact_vs_native_agreement": float(np.mean([r["exact_argmax"] == native[i] for i, r in enumerate(rows)])),
            "headroom_quantiles": {q: float(np.quantile(head, q)) for q in (0.05, 0.25, 0.5, 0.75, 0.95)},
            "err_inf_median": float(np.median([r["err_inf"] for r in rows])),
            "err_l2_median": float(np.median([r["err_l2"] for r in rows])),
            "ref_margin12_median": float(np.median([r["margin12"] for r in rows])),
        }
        n = len(rows) // 2
        summary["by_source"] = {"wiki": float(np.mean(cert[:n])), "gsm8k": float(np.mean(cert[n:]))}
        report["runs"][f"drop{k}"] = {"summary": summary, "rows": rows}
        print(f"drop {k}: " + json.dumps(summary), flush=True)
    report["seconds"] = time.time() - t0
    with open("results_rtn.json" if ROUND else "results.json", "w") as f:
        json.dump(report, f, indent=1)


if __name__ == "__main__":
    main()
