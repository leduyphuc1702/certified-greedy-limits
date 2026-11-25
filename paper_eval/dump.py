"""Stage 1 (MPS): held-out + calibration hidden states, native argmax, coarse
hidden states for the v20 oracle test, LM head bits. One model per run."""
import sys

import numpy as np
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

PROMPT_TOKENS = 48
GEN = 8
PROJ = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")
DEV = "mps"


def wiki(split, lo, hi):
    out = []
    for row in load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split=split):
        s = row["text"].strip()
        if s and not s.startswith("=") and len(s.split()) >= 40:
            out.append(s)
        if len(out) == hi:
            break
    return out[lo:hi]


def prompts(tok, wiki_rows, gsm_rows):
    out = [tok(s)["input_ids"][:PROMPT_TOKENS] for s in wiki_rows]
    for q in gsm_rows:
        ids = tok.apply_chat_template([{"role": "user", "content": q}], add_generation_prompt=True)
        if not isinstance(ids, list):
            ids = ids["input_ids"]
        out.append(ids[-PROMPT_TOKENS:])
    return out


def load(name, drop=0, rnd=False):
    m = AutoModelForCausalLM.from_pretrained(name, dtype=torch.bfloat16).to(DEV).eval()
    if drop:
        mask = ~((1 << drop) - 1)
        with torch.no_grad():
            for n, mod in m.named_modules():
                if n.split(".")[-1] in PROJ:
                    b = mod.weight.data.view(torch.int16)
                    if rnd:
                        b = b + (1 << (drop - 1))
                    mod.weight.data.copy_((b & mask).view(torch.bfloat16))
    return m


@torch.no_grad()
def generate(m, ps):
    seqs = []
    for ids in ps:
        x = torch.tensor([ids], device=DEV)
        g = m.generate(x, attention_mask=torch.ones_like(x), max_new_tokens=GEN, min_new_tokens=GEN, do_sample=False)
        seqs.append((g[0].tolist(), len(ids)))
    return seqs


@torch.no_grad()
def hidden(m, seqs):
    hs, am = [], []
    for ids, p in seqs:
        o = m(torch.tensor([ids], device=DEV), output_hidden_states=True)
        sl = slice(p - 1, p - 1 + GEN)
        hs.append(o.hidden_states[-1][0, sl].float().cpu().numpy())
        am.append(o.logits[0, sl].float().argmax(-1).cpu().numpy())
    return np.concatenate(hs), np.concatenate(am)


def main():
    name, tag = sys.argv[1], sys.argv[2]
    tok = AutoTokenizer.from_pretrained(name)
    gsm_test = load_dataset("openai/gsm8k", "main", split="test")["question"]
    gsm_train = load_dataset("openai/gsm8k", "main", split="train")["question"]
    held = prompts(tok, wiki("test", 16, 80), gsm_test[16:80])
    cal = prompts(tok, wiki("validation", 0, 16), gsm_train[0:16])
    ref = load(name)
    seqs = generate(ref, held)
    h, native = hidden(ref, seqs)
    h_cal, _ = hidden(ref, generate(ref, cal))
    W = ref.get_output_embeddings().weight.detach().cpu().view(torch.int16).numpy()
    del ref
    torch.mps.empty_cache()
    coarse = {}
    for rnd in (False, True):
        m = load(name, 4, rnd)
        coarse["rtn" if rnd else "trunc"] = hidden(m, seqs)[0]
        del m
        torch.mps.empty_cache()
    np.savez(f"{tag}.npz", h=h, native=native, h_cal=h_cal, W_bits=W, h_trunc=coarse["trunc"], h_rtn=coarse["rtn"])
    print(tag, h.shape, W.shape)


if __name__ == "__main__":
    main()
