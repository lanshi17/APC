# Reproduction package: *Validation Argmax Is Hard to Beat*

This directory contains the artifacts for the TMLR submission, including the
P3 extended corpus (Scenarios A, B, C, §5.6 of the paper).

| File | What it is |
|---|---|
| `negative-result-tmlr.pdf` | the submission (TMLR style, 15 pages) |
| `negative-result-tmlr.tex` | its LaTeX source (generated from the markdown) |
| `references.bib` | the nine references cited in the paper |
| `tmlr.bst` | the TMLR bibliography style used to typeset them |
| `tmlr.sty` | the TMLR template style (with the four `\@ifundefined` guards that TeX Live's `jmlr.cls` needs) |
| `negative-result-tmlr.bbl` | the generated bibliography, so the `.tex` also compiles without BibTeX |
| `supplementary.pdf` | supplementary material: protocols, per-group tables, diagnostics, win profile (7 pages) |
| `code.zip` | anonymised code bundle (`git archive` of the code repo HEAD, identity-scanned) |
| `README_reproduce.md` | this file |

To rebuild the submission from source, with `figures/` alongside it:

```bash
pdflatex negative-result-tmlr && bibtex negative-result-tmlr \
  && pdflatex negative-result-tmlr && pdflatex negative-result-tmlr
```

`jmlr.cls` declares `plainnat.bst` in the preamble, so a first `pdflatex` pass
writes two `\bibstyle` lines into the `.aux`. BibTeX rejects the second one
(*Illegal, another \bibstyle command*), falls back to `plainnat.bst`, and every
citation comes out as `(?)`. Delete the extra `\bibstyle{plainnat}` line before
running BibTeX -- that is what `docs/paper_patch/make_tmlr.py` does.

The authoritative sources live in two sibling repositories, which is why the
review package ships a snapshot rather than a single tree:

* **code** — `02_APC/` (this bundle): `jev_rsi/` package, frozen + extended
  corpora, all result JSON artifacts, `scripts/p3_acceptance.py`.
* **paper** — `02_apc_paper/`: manuscript markdown (`manuscript_negative/negative-result.md`),
  `build.py` (markdown → LaTeX → PDF), and the paper-side analysis docs.

## 1. Environment

Python 3.13. `code.zip` ships a `pyproject.toml`; the runs below need:

```bash
python -m venv .venv && . .venv/bin/activate
pip install -e '.[dev]'          # numpy, scipy, torch (CPU), transformers, bert-score
```

`ROUGE-L` is pure Python; `BERTScore` uses `microsoft/deberta-xlarge-mnli`
(downloaded on first use, ~1.6 GB, cached under `$HF_HOME`).

No API key is needed to reproduce any number in the paper:

* the **frozen corpus** (43 groups) is a replay from committed validation
  scores — deterministic, no network;
* every **extended-corpus** model call (generation, judging, embedding) is
  response-cached on disk under `jev_rsi/corpus_extended/_cache/`, keyed by
  model, messages, temperature, `max_tokens`, seed and tag, so a re-run is free
  and byte-identical. A DashScope key is only needed to *extend* the corpus,
  not to reproduce it.

## 2. Reproduce the paper

```bash
cd code                                     # the unpacked bundle

# frozen corpus (43 groups / 32 decisions): 94 seconds, byte-identical
python -m jev_rsi.experiments

# extended corpus (88 groups / 485 arms / 77 decisions)
python -m jev_rsi.experiments --corpus extended
python -m jev_rsi.analysis_extended

# pre-registered acceptance criteria (10 pass / 1 partial / 1 fail)
python scripts/p3_acceptance.py

# every number quoted in the manuscript, checked against the JSON artifacts
python docs/paper_patch/verify_numbers.py --md ../02_apc_paper/manuscript_negative/negative-result.md
```

Built artifacts (all byte-reproducible, `git status` stays clean):
`jev_rsi/results/jev_rsi_results.json` (frozen),
`jev_rsi/results/jev_rsi_extended_results.json`,
`jev_rsi/results/jev_rsi_extended_analysis.json`.

Regenerating the scenario corpora themselves (needs a DashScope key; cached
calls are skipped, so on the released cache this is also free):

```bash
python -m jev_rsi.corpus_extended.fetch_sources cnn_dailymail --n 10
python -m jev_rsi.corpus_extended.fetch_sources flickr30k --n 15
python -m jev_rsi.corpus_extended.build_scenario_c     # adversarial summaries, 10 x 6
python -m jev_rsi.corpus_extended.build_scenario_a     # creative writing, 20 x 10
python -m jev_rsi.corpus_extended.build_scenario_b     # image captions, 15 x 6
python -m jev_rsi.corpus_extended.probe_judge_variance \
    --models qwen3.8-flash qwen3.8-27b --groups 5 --draws 6
```

## 3. Rebuild the documents

```bash
cd ../02_apc_paper/manuscript_negative
python build.py                     # markdown -> negative-result.tex -> .pdf (15 pages)
python scripts/check_reproducibility.py   # re-runs the suite, asserts git stays clean
                                         # --quick = frozen corpus and its five modules
python ../code/docs/paper_patch/make_supplementary.py       # -> supplementary.pdf
python ../code/docs/paper_patch/make_tmlr.py                # -> tmlr/negative-result-tmlr.pdf
```

`docs/paper_patch/` in the code bundle holds the three generators plus
`patch_manuscript.py`, the idempotent script that inserted the §5.6 results,
abstract/Introduction/Conclusion updates and the Discussion paragraph into the
manuscript markdown.

## 4. Models and deviations (summary)

* The plan named `gpt-4o` / `gpt-4o-vision`; no key was available. Every text
  was generated and judged by `qwen3.8-flash`, with `qwen3.8-27b` as an
  independent judge for the variance probe. Both are documented in the corpus
  provenance blocks.
* Summaries are scored against the dataset's **human highlights** (ROUGE-L F1)
  rather than the raw article; the article-referenced variants are recorded too.
* Holdout BERTScore is **baseline-rescaled**, i.e. it measures gain over a
  baseline extractive summary, because raw F1 is ~0.9 for any fluent text.
* All judge scores are normalised 0–1 for the gate (raw 0–10 kept alongside).
* Scenario B adds a vision **ranking** holdout because the 0–10 vision score
  saturates.
* Judge calls keep the model's reasoning chain (~500 completion tokens each);
  generation calls disable it. A token ledger with per-call usage and cost is at
  `jev_rsi/corpus_extended/_cache/usage.jsonl`.
