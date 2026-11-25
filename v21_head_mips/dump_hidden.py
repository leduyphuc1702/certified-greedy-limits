"""Save final-norm hidden states: the 256 v20 test positions and an independent
calibration set (WikiText-2 validation + GSM8K train), plus the LM head."""
import sys

import numpy as np
import torch
from datasets import load_dataset
from transformers import AutoTokenizer

sys.path.insert(0, "../v20_killtest")
import oracle_box as ob  # noqa: E402


def calib_prompts(tok):
    out = []
    for row in load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="validation"):
        s = row["text"].strip()
        if s and not s.startswith("=") and len(s.split()) >= 40:
            out.append(tok(s)["input_ids"][: ob.PROMPT_TOKENS])
        if len(out) == 16:
            break
    for row in load_dataset("openai/gsm8k", "main", split="train").select(range(16)):
        ids = tok.apply_chat_template([{"role": "user", "content": row["question"]}], add_generation_prompt=True)
        if not isinstance(ids, list):
            ids = ids["input_ids"]
        out.append(ids[-ob.PROMPT_TOKENS:])
    return out


def run(ref, prompt_ids):
    seqs = []
    for ids in prompt_ids:
        x = torch.tensor([ids], device=ob.DEV)
        g = ref.generate(x, attention_mask=torch.ones_like(x), max_new_tokens=ob.GEN, min_new_tokens=ob.GEN, do_sample=False)
        seqs.append((g[0].tolist(), len(ids)))
    return ob.final_hidden(ref, seqs)


def main():
    tok = AutoTokenizer.from_pretrained(ob.MODEL)
    ref = ob.load(0)
    h_test, native_test = run(ref, [ids for _, ids in ob.prompts(tok)])
    h_cal, _ = run(ref, calib_prompts(tok))
    W = ref.lm_head.weight.detach().cpu().view(torch.int16).numpy()
    np.savez("hidden.npz", h_test=h_test, native_test=native_test, h_cal=h_cal, W_bf16_bits=W)
    print(h_test.shape, h_cal.shape, W.shape)


if __name__ == "__main__":
    main()
