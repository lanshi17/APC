#!/usr/bin/env python3
"""Append the P3 extended-corpus sections to the paper-side docs.

    python docs/paper_patch/update_paper_docs.py [--paper DIR] [--dry-run]

Targets ``RESULTS_ANALYSIS.md``, ``EXPERIMENT_PROTOCOL.md`` and
``README_JEV_RSI.md`` in the paper repository.  Idempotent: each block carries a
marker comment and is written at most once.
"""
from __future__ import annotations

import argparse
import os
import pathlib
import sys

CODE = pathlib.Path(__file__).resolve().parents[2]
PAPER = pathlib.Path(os.environ.get("APC_PAPER_DIR",
                                    CODE.parent / "02_apc_paper"))
MARK = "<!-- P3-EXTENDED-CORPUS -->"

RESULTS = f"""
---

{MARK}
## 12. P3 Extended Corpus: Scenarios A, B, C

Section 5.6 of the manuscript (`manuscript_negative/negative-result.md`) adds 45
decision groups (350 arms) whose purpose was to give the structured gate an
advantage. Corpus total: **88 groups / 485 arms / 77 deployment decisions**.

### 12.1 Deployment outcomes (77 decision groups)

| Policy | Exact oracle | Within noise | Avg regret |
|---|---|---|---|
| `greedy` (val-argmax) | **47/77 (61.0%)** | 56/77 | +0.01754 |
| `heuristic` | 42/77 (54.5%) | 55/77 | +0.01773 |
| `jev` (static gate) | 37/77 (48.1%) | 51/77 | +0.05184 |

Paired `greedy − jev_static` = −0.0343, 95% CI [−0.0658, −0.0082],
W/L/T = 7/21/49, sign test p = 0.0125 — a larger and more robust deficit than
the frozen corpus's (32 groups, −0.0013, p = .035).

### 12.2 Per-scenario results

| Scenario | n | `greedy` | `jev` | Why structure lost |
|---|---|---|---|---|
| C: adversarial summarisation | 10 | 6/10 | 6/10 | ROUGE-L is not gameable: gaming arms score 0.121–0.171 vs 0.252 for honest abstractive; val max 0.380 < the pre-registered 0.4 capture band; 0/10 groups captured (also 0/10 under the other 7 recorded metrics) |
| A: high-variance judge | 20 | 8/20 | 5/20 | The judge is not noisy (σ̄ = 0.47 of 10; 0/20 groups, 0/100 probed arms above σ = 2.0). The gate still retreats on 15/20 groups to `zero_shot`, which is not a safe arm here (+.1655 vs +.0355 regret) |
| B: text vs vision judge | 15 | 15/15 | 15/15 | The two judges agree: Spearman(val, vision) = +.346, and +.527 against a non-saturating ranking holdout. Vision scores sit at the ceiling (5/6 styles ≥ 0.86) |

### 12.3 Mechanism: scale migration, not a broken primitive

The gate's three thresholds are constants calibrated on the frozen corpus.
Median top-1/top-2 validation margin, in units of `NOISE_BAND` = 0.007:

| Corpus | median margin | × NOISE_BAND | alignment rule saturated |
|---|---|---|---|
| frozen | 0.0006 | 0.1× | 28% |
| C | 0.0257 | 3.7× | 60% |
| A | 0.0000 | 0.0× | 35% |
| B | 0.1000 | 14.3× | 67% |

Refitting the noise band per scenario and re-running the gate's own
self-improvement learner returns Scenario C to parity with validation argmax
(6/10, +.0239) while the statically calibrated gate gets worse (4/10, +.0435).
Scenario A does not recover, because no threshold makes an unsafe fallback safe.

**Artifacts**: `code/jev_rsi/results/jev_rsi_extended_results.json`,
`..._analysis.json`; corpora and per-call token ledger under
`code/jev_rsi/corpus_extended/`; acceptance report from
`code/scripts/p3_acceptance.py` (10 pass / 1 partial / 1 fail, where the FAIL is
the pre-registered judge-variance premise that measurement disproved).
"""

PROTOCOL = f"""
---

{MARK}
## 5. P3 Extended Corpus: Protocol (post-registration, 2025-01-04)

The frozen protocol in §1–§3 is unchanged.  P3 adds three scenarios, each with a
pre-registered premise and a pre-registered falsification band; the premises
were registered before the judge outputs were inspected.

| Scenario | Construction | Validation (deployment) | Holdout | Pre-registered premise |
|---|---|---|---|---|
| C | 10 CNN/DailyMail articles × 6 summaries (lead-3, TF-IDF extractive, abstractive, template, keyword-stuffed, random-highlight) | ROUGE-L F1 vs human highlights | baseline-rescaled BERTScore F1 (deberta-xlarge-mnli) vs the same highlights | ≥15 arms with val > 0.4 **and** holdout < 0.3; val-argmax captured by a gaming arm |
| A | 20 creative-writing tasks × 10 prompt strategies (Occam-ordered, `zero_shot` is the designated safe arm) | one judge draw | mean of 5 further draws, temperature 1.0 | ≥5 groups with per-arm judge σ > 2.0 (0–10) |
| B | 15 Flickr30k images × 6 caption styles | text-only judge vs 5 human references | vision judge (image + caption), plus a within-image strict ranking of the 6 captions | text judge disagrees with vision judge; vision judge discriminates styles |

**Observed protocol deviations** (all recorded in the provenance blocks):

1. `gpt-4o`/`gpt-4o-vision` unavailable → `qwen3.8-flash` for all text, with
   `qwen3.8-27b` used only for the independent variance probe.
2. Summary scoring switched from "vs full article" to "vs human highlights",
   because the article-referenced ROUGE-L F1 caps near 0.2 for every arm and
   makes the registered capture band unreachable for reasons unrelated to
   gaming. Both variants are recorded for every arm.
3. BERTScore reported baseline-rescaled (raw F1 is ~0.9 for any fluent text).
4. All judge scores normalised 0–10 → 0–1 for the gate; raw values retained.
5. Scenario B gained a strict-ranking holdout after the 0–10 vision score
   saturated; the ranking is used for the agreement claim.
6. The `extractive` arm is a deterministic TF-IDF selector, not an LLM call.

**Determinism and cost**: every model response is cached on disk and every call
is logged to `corpus_extended/_cache/usage.jsonl`. Re-running any builder hits
the cache and costs nothing; the frozen artifact is untouched
(`python -m jev_rsi.experiments --corpus frozen` is byte-identical).
"""

README = f"""
---

{MARK}
## 🧩 P3 Extended Corpus (2025-01-04)

The corpus grew from 43 to **88 groups / 485 arms / 77 decision groups** with
three purpose-built scenarios (adversarial summarisation, high-variance judge,
text-vs-vision captioning). All three pre-registered premises failed to hold on
real model runs, and the gate's deficit grew: **37/77 vs 47/77** for validation
argmax (paired p = .0125, 7 wins / 21 losses / 49 ties). The residual losses are
explained by threshold scale migration (0.1×–14.3× `NOISE_BAND`) and by an
unsafe fallback arm, not by missing the intended regime.

```bash
cd code
python -m jev_rsi.experiments --corpus extended     # 77 decision groups
python -m jev_rsi.analysis_extended                  # scenario tables + calibration
python scripts/p3_acceptance.py                      # 10 pass / 1 partial / 1 fail
```

* Manuscript: §5.6 of `manuscript_negative/negative-result.md`
  (also `supplementary.pdf`, `tmlr/negative-result-tmlr.pdf`)
* Results narrative: [RESULTS_ANALYSIS.md](RESULTS_ANALYSIS.md) §12
* Protocol and deviations: [EXPERIMENT_PROTOCOL.md](EXPERIMENT_PROTOCOL.md) §5
* Task tracking: [P3_TASK_TRACKER.md](P3_TASK_TRACKER.md)
"""

TARGETS = [("RESULTS_ANALYSIS.md", RESULTS),
           ("EXPERIMENT_PROTOCOL.md", PROTOCOL),
           ("README_JEV_RSI.md", README)]

def _note_results() -> str:
    return ("*(Frozen corpus. Full-corpus totals over the 77 decision groups of the "
            "P3 extension are in \u00a712 below.)*")


def _note_readme() -> str:
    return ("*(Frozen corpus. Full-corpus totals over the 77 decision groups of the "
            "P3 extension are in the *P3 Extended Corpus* section below.)*")


_OLD_NOTE = ("*(Frozen corpus. Full-corpus totals over the 77 decision groups of the "
             "P3 extension are in \u00a710 below.)*")

# Surgical fixes: pointers so the frozen-corpus tables are not mistaken for the
# whole corpus, plus the renumbering of the appended block (the file already used
# 10 and 11 before the P3 section, so the append must not reuse §10).  Entries are
# applied in order, at most once each: the new text already being present (or, for
# a deletion, the old text being absent) means "done".
_H3_RESULTS = "### 3.1 Performance Comparison (32 Decision Groups)"
_H3_README = "### Main Results (32 Decision Groups)"
POST_EDITS = [
    # repair an already-patched file: heading -> heading + stale note
    ("RESULTS_ANALYSIS.md", _H3_RESULTS + "\n\n" + _OLD_NOTE,
     _H3_RESULTS + "\n\n" + _note_results()),
    ("README_JEV_RSI.md", _H3_README + "\n\n" + _OLD_NOTE,
     _H3_README + "\n\n" + _note_readme()),
    # fresh file: heading -> heading + note
    ("RESULTS_ANALYSIS.md", _H3_RESULTS, _H3_RESULTS + "\n\n" + _note_results()),
    ("README_JEV_RSI.md", _H3_README, _H3_README + "\n\n" + _note_readme()),
    # renumber the appended block
    ("RESULTS_ANALYSIS.md", "## 10. P3 Extended Corpus: Scenarios A, B, C",
     "## 12. P3 Extended Corpus: Scenarios A, B, C"),
    ("RESULTS_ANALYSIS.md", "### 10.1 Deployment outcomes (77 decision groups)",
     "### 12.1 Deployment outcomes (77 decision groups)"),
    ("RESULTS_ANALYSIS.md", "### 10.2 Per-scenario results",
     "### 12.2 Per-scenario results"),
    ("RESULTS_ANALYSIS.md", "### 10.3 Mechanism: scale migration, not a broken primitive",
     "### 12.3 Mechanism: scale migration, not a broken primitive"),
    # drop any stale note that survived an earlier round
    ("RESULTS_ANALYSIS.md", _OLD_NOTE, ""),
    ("README_JEV_RSI.md", _OLD_NOTE, ""),
    ("README_JEV_RSI.md", "[RESULTS_ANALYSIS.md](RESULTS_ANALYSIS.md) \u00a710",
     "[RESULTS_ANALYSIS.md](RESULTS_ANALYSIS.md) \u00a712"),
]


def apply_pointers(root: pathlib.Path) -> list[str]:
    log = []
    for name, old, new in POST_EDITS:
        p = root / name
        tag = (new or f"drop: {old}").splitlines()[-1][:46]
        if not p.exists():
            log.append(f"skip (missing): {name}")
            continue
        s = p.read_text(encoding="utf-8")
        if (old not in s) if not new else (new in s):
            log.append(f"already applied: {name} :: {tag}")
            continue
        if s.count(old) != 1:
            log.append(f"skip (anchor not unique, {s.count(old)}x): {name} :: {tag}")
            continue
        p.write_text(s.replace(old, new, 1), encoding="utf-8")
        log.append(f"applied: {name} :: {tag}")
    return log


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--paper", default=str(PAPER))
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    root = pathlib.Path(args.paper)
    for name, block in TARGETS:
        p = root / name
        if not p.exists():
            print(f"skip (missing): {p}", file=sys.stderr)
            continue
        s = p.read_text(encoding="utf-8")
        if MARK in s:
            print(f"already updated: {name}")
            continue
        if args.dry_run:
            print(f"would append {len(block)} chars to {name}")
            continue
        p.write_text(s.rstrip("\n") + "\n" + block, encoding="utf-8")
        print(f"updated: {name} (+{len(block)} chars)")
    for line in apply_pointers(root):
        print(line)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
