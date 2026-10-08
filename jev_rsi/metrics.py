"""Scoring metrics for the P3 extended corpus.

Two families are used, deliberately of different *kinds* so that a val/holdout
decoupling is meaningful:

* lexical overlap  -- ``rouge_score`` (Google's ROUGE-1/R-2/R-L)  -> val side
* semantic overlap -- ``bert_score`` (contextual embeddings)      -> holdout side

Both take the **human-written reference** as the gold target:

* Scenario C: the CNN/DailyMail gold highlights (the dataset's own human
  summary).  The P3 plan's inline pseudo-code scored summaries against the full
  *article*; that is not the standard protocol and collapses the scale (an LCS
  against a 10x longer document caps ROUGE-L F1 near 0.15 for every arm, so no
  arm can be separated).  We therefore score against the gold highlights and
  *additionally* record the article-referenced variant for transparency.
* Scenario B: the Flickr30k reference captions.

``bert_score`` is loaded lazily: importing it pulls torch, which the rest of the
harness does not need.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

# Default BERTScore model = the one named in the P3 plan.
BERTSCORE_MODEL = "microsoft/deberta-xlarge-mnli"


def rouge(hyp: str, ref: str) -> Dict[str, float]:
    """ROUGE-1/2/L F1 (stemmed, Google rouge_score)."""
    from rouge_score import rouge_scorer
    sc = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    s = sc.score(ref, hyp)
    return {"rouge1": round(s["rouge1"].fmeasure, 6),
            "rouge2": round(s["rouge2"].fmeasure, 6),
            "rougeL": round(s["rougeL"].fmeasure, 6)}


def bertscore(hyps: Sequence[str], refs: Sequence[str],
              model_type: str = BERTSCORE_MODEL,
              batch_size: int = 8, raw: bool = False) -> List[float]:
    """BERTScore F1 per pair, using the plan's ``deberta-xlarge-mnli``.

    ``raw=False`` (default) returns the **baseline-rescaled** F1, i.e. the
    standard interpretable BERTScore where an unrelated candidate scores ~0 and
    a human-paraphrase-quality candidate scores ~0.6-0.8.  The P3 plan's
    acceptance band (``holdout < 0.3``) only makes sense on that scale: raw
    embedding-cosine F1 sits at 0.85-0.95 for *any* grammatical text, so no arm
    could ever fall below 0.3.  ``raw=True`` returns the un-rescaled F1.
    """
    from bert_score import score as _score
    _, _, F = _score(list(hyps), list(refs), lang="en", model_type=model_type,
                     batch_size=batch_size, verbose=False,
                     rescale_with_baseline=not raw)
    return [round(float(x), 6) for x in F]


def bertscore_both(hyps: Sequence[str], refs: Sequence[str],
                   model_type: str = BERTSCORE_MODEL,
                   batch_size: int = 8) -> Dict[str, List[float]]:
    """Raw and baseline-rescaled F1 from one process (the model is lru_cached)."""
    return {"f_rescaled": bertscore(hyps, refs, model_type, batch_size, raw=False),
            "f_raw": bertscore(hyps, refs, model_type, batch_size, raw=True)}


def pearson(xs: Sequence[float], ys: Sequence[float]) -> float:
    n = len(xs)
    if n < 2:
        return 0.0
    mx, my = sum(xs) / n, sum(ys) / n
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    dx = sum((x - mx) ** 2 for x in xs) ** 0.5
    dy = sum((y - my) ** 2 for y in ys) ** 0.5
    return num / (dx * dy) if dx and dy else 0.0


def spearman(xs: Sequence[float], ys: Sequence[float]) -> float:
    def ranks(v: Sequence[float]) -> List[float]:
        order = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and v[order[j + 1]] == v[order[i]]:
                j += 1
            avg = (i + j) / 2 + 1
            for k in range(i, j + 1):
                r[order[k]] = avg
            i = j + 1
        return r
    return pearson(ranks(xs), ranks(ys))
