# Feasibility Limits of Certified Low-Precision Greedy Decoding

Code, preregistrations and raw results for the preprint
*Feasibility Limits of Certified Low-Precision Greedy Decoding: A Preregistered Study with an Oracle Ceiling* (Phuc Le, 2026). The LaTeX source is in `paper/`.

## Main results (held-out, 1,024 positions per model)

| Test | Pass threshold | Qwen2.5-0.5B | TinyLlama-1.1B |
|---|---|---|---|
| v20 oracle box, 4 bits truncated | cert. rate ≥ 95 % | 4.7 % | 12.1 % |
| v20 oracle box, 4 bits rounded | cert. rate ≥ 95 % | 18.6 % | 26.5 % |
| v21a exact MIPS tree | mean ρ ≤ 15 % | 110.9 % | 106.5 % |
| v21b dimension-adaptive | mean ρ ≤ 25 % | 82.6 % | 99.8 % |
| v21c SVD basis | mean ρ ≤ 25 % | 97.9 % | 98.8 % |
| v22 certified head, BF16 out | refine ≤ 5 % | 7.2 % | 7.3 % |
| v23 certified head, FP32 out | refine ≤ 1 % | 0.68 % | 1.76 % |

ρ = logical bytes read / bytes of one dense LM-head pass. All runs: zero false certificates, zero wrong tokens.

## Layout

| Directory | Contents |
|---|---|
| `v20_killtest/` | Development oracle-box test (256 Qwen positions). `GITI_ROUND=1` switches to round-to-nearest. |
| `v21_head_mips/` | Development hidden-state dump and the v21a tree / v21b dimension-adaptive search code (imported by `paper_eval`). |
| `v22_certified_head/` | Development certified BF16-out head. |
| `paper_eval/` | Held-out evaluation behind Table 1 of the paper, oracle audit and reference audit. |
| `paper/` | LaTeX source (`tectonic main.tex`). |

Each directory has a `PREREG.md` with the pass thresholds, written before its run. These files were kept privately and not deposited in a public registry before the experiments; they are released verbatim.

## Reproduce

Apple Silicon with 16 GB RAM, Python 3.11.

```bash
python -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
cd paper_eval
python dump.py Qwen/Qwen2.5-0.5B-Instruct qwen          # MPS, writes qwen.npz (~280 MB)
python dump.py TinyLlama/TinyLlama-1.1B-Chat-v1.0 tinyllama
python evaluate.py qwen && python evaluate.py tinyllama  # CPU FP64, ~20 min per model
python oracle_audit.py qwen && python oracle_audit.py tinyllama
python audit_reference.py
```

Run one heavy job at a time; on a 16 GB machine `OMP_NUM_THREADS=4 VECLIB_MAXIMUM_THREADS=4` avoids swapping.
Development runs: `v20_killtest/oracle_box.py`, then `v21_head_mips/dump_hidden.py` and `killtests.py`, then `v22_certified_head/certified_head.py`.

Environment used: macOS 27.0, Apple M4, PyTorch 2.15.0, Transformers 5.24.1, NumPy 2.4.6, Datasets 5.2.0. Model revisions: Qwen2.5-0.5B-Instruct `7ae5576`, TinyLlama-1.1B-Chat-v1.0 `fe8a4ea`.

## License

Code: MIT. Paper text: CC BY 4.0.
