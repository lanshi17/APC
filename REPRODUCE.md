# Reproducing the APC Paper Experiments

All numbers in `docs/paper-apc.md` (Chinese master) / `docs/paper-apc-en.md` are
traceable to `experiments/apcbench/*.json`. Two tiers:

## Tier 0 — deterministic, no network, no credentials

```bash
python -m venv .venv && .venv/bin/pip install -e apc-pipeline
cd apc-pipeline && ../.venv/bin/python -m pytest tests -q     # 41 unit tests
../.venv/bin/python ../scripts/analyze_bench.py                # simulation meta-analysis
cd .. && .venv/bin/python scripts/test_external_judge.py       # judge v3.2 regression (24 checks)
```

Simulation benchmarks (`bench_apc.py`, `bench_contract.py`, `bench_math.py`,
`bench_follow.py`, `bench_transfer.py`, `eval_robustness.py`) run against the built-in
mock model pool (three profiles incl. a simulated frontier reasoner) with fixed seeds —
fully offline and bit-stable; F1-F5/F10-simulation numbers come from this tier.

## Tier 1 — real-API experiments (need one OpenAI-compatible credential)

Configure via `.env` (gitignored; never commit keys):

```bash
APC_QWEN_MODEL=<whitelisted model id>     # e.g. qwen3.8-flash
APC_QWEN_BASE_URL=<https://.../v1>
APC_QWEN_API_KEY=<sk-...>
```

Run order (each writes its own `experiments/apcbench/real_*.json`; reruns are
idempotent — the `--seed`/`--part` keys dedupe rows):

```bash
.venv/bin/python scripts/bench_real_full.py                   # F1-F5: zero-shot floor, arms, transfer, robustness
.venv/bin/python scripts/bench_real_external.py               # F6: GSM8K / MATH-L5 / AIME (hardened judge v3.2)
.venv/bin/python scripts/bench_real_gepa.py                   # F7: official GEPA baseline + paired re-measure
.venv/bin/python scripts/bench_real_espo.py                   # F9: ESPO reproduction (Diagnose/Propose/Select)
.venv/bin/python scripts/auto_apc_gate.py                     # F8: zero-cost deployment gate (replay + blind lookahead)
.venv/bin/python scripts/bench_real_reeval.py --task financial --max-tokens 150 --gepa-champ <file>   # F10 arms; see paper §3.4 for the six-pathway paired batch
# F11: HLE contest (pre-reg docs/literature/F11-prereg.md). Dataset already in-repo:
#   datasets/hle_exact/{dev,validation,holdout}.jsonl  (regenerate: scripts/gen_hle_dataset.py,
#   upstream cais/hle via ungated mirror, MIT; stratified 30/30/30, seed 2026)
.venv/bin/python scripts/bench_real_hle.py --arms z0,champs --hold-n 30 --seed 921 --timeout 600 --hard 640
.venv/bin/python scripts/bench_real_hle.py --arms gepa,espo  --hold-n 30 --seed 921 --timeout 600 --hard 640
```

Real-API notes learned the hard way (all implemented in the scripts):
- **Pair same-day.** temp=0 reasoning APIs drift ±.007 within a day and ~.028 across
  days; every verdict in §3.4 uses paired same-day bands (F7 protocol).
- **Resume after network faults.** `bench_real_hle.py` checkpoints each sample to
  `/tmp/hle_{arm}_{seed}.jsonl`; before resuming, delete rows whose `pred` contains
  `<call-error>` (they count as done otherwise).
- **Long runs: no nohup.** Use a supervised process (the repo runs them under an
  agent-hub supervisor with restart=no); detached background shells lose oversight
  and have died silently before.
- Connection-layer reliability (F11): stateless per-request HTTP
  (`_fresh_http_call`), thread-level hard timeouts (`--hard 640`), per-sample streams.

## Multi-model extension

`bench_real_full.py --model <id>` with three extra `.env` lines
(`APC_<ID>_MODEL/BASE_URL/API_KEY`) runs the whole matrix on a new model with zero
code changes. PGAM validation (paper §4 #3) and the weak-model + protocol-pressure
cell (§4 #5) are gated on additional credentials.

## Anonymized submission bundle

`bash scripts/make_anon_bundle.sh` → `/tmp/apc-anon-bundle.zip`: git-HEAD snapshot
with identity files stripped (LICENSE/frontier notes), README de-signed, and hard
gates that abort on any residual author path/id or credential-shaped token
(word-boundary `sk-…{16,}` scan). Run it before uploading supplementary material.
