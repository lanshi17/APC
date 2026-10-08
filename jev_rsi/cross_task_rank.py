"""P6 -- cross-task source selection as ranking rather than regression.

``FUTURE_DIRECTIONS.md`` Priority 6 observes that the regression formulation
scores 1/3 on choosing a transfer source and asks for a ranking formulation
instead, with a target of at least 2/3.

This module runs that comparison on the same 12 cells and 6 targets, with
leave-target-out training throughout, and reports selection *regret* alongside
accuracy -- on this corpus the two disagree, because one of the three
multi-candidate targets has an oracle margin of only 0.0001, so an exact-hit
accuracy can be lost on a difference four orders of magnitude below the
validation noise band.

It also states, and then verifies, a structural limit that turns out to explain
the result: within a target every candidate shares the same target-level
features, so only source-level features can separate candidates, and a scorer
built from source-level features alone is constant across targets.  Such a
ranker is therefore equivalent to always choosing one fixed source.

Usage
-----
    python -m jev_rsi.cross_task_rank
    python -m jev_rsi.cross_task_rank --out /tmp/x
"""
from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .cross_task import (CHOICE_ORDER, NONE, SOURCE_ORDER, TransferCell,
                         _actual, _candidates, _choose, _targets, _train_pool,
                         extract_cells, predict)
from .data import load_groups

DEFAULT_OUT = Path(__file__).resolve().parent / "results"
FEATURES = ("s_mean", "s_pos_rate", "s_best_rate", "is_none",
            "s_champion_mean", "s_warm_mean")


# ---------------------------------------------------------------------------
# features
# ---------------------------------------------------------------------------
def _source_features(pool: Sequence[TransferCell], target: str,
                     candidates: Sequence[str]) -> Dict[str, Dict[str, float]]:
    """Source-level features for each candidate, computed on the training pool."""
    gm = float(np.mean([c.gain for c in pool])) if pool else 0.0
    pool_targets = sorted({c.target for c in pool})
    # per-target argmax over the pool, to get a "best source rate"
    best_of: Dict[str, str] = {}
    for t in pool_targets:
        act = _actual(pool, t)
        cands = _candidates(pool, t)
        best_of[t] = _choose(act, cands)

    out: Dict[str, Dict[str, float]] = {}
    for s in candidates:
        if s == NONE:
            out[s] = {f: 0.0 for f in FEATURES}
            out[s]["is_none"] = 1.0
            continue
        obs = [c for c in pool if c.source == s]
        gains = [c.gain for c in obs]
        out[s] = {
            "s_mean": float(np.mean(gains)) if gains else gm,
            "s_pos_rate": (sum(1 for x in gains if x > 0) / len(gains)) if gains else 0.0,
            "s_best_rate": (sum(1 for t in pool_targets if best_of[t] == s)
                            / len(pool_targets)) if pool_targets else 0.0,
            "is_none": 0.0,
            "s_champion_mean": (float(np.mean([c.gain for c in obs if c.kind == "champion"]))
                                if any(c.kind == "champion" for c in obs) else gm),
            "s_warm_mean": (float(np.mean([c.gain for c in obs if c.kind == "warm-start"]))
                            if any(c.kind == "warm-start" for c in obs) else gm),
        }
    return out


# ---------------------------------------------------------------------------
# pairwise logistic regression (Bradley-Terry), no sklearn dependency
# ---------------------------------------------------------------------------
def _fit_pairwise(X: np.ndarray, y: np.ndarray, l2: float = 0.05,
                  lr: float = 0.2, iters: int = 4000) -> np.ndarray:
    w = np.zeros(X.shape[1])
    for _ in range(iters):
        z = X @ w
        p = 1.0 / (1.0 + np.exp(-z))
        grad = X.T @ (p - y) / len(y) + l2 * w
        w -= lr * grad
    return w


def _pairwise_data(pool: Sequence[TransferCell]) -> Tuple[np.ndarray, np.ndarray]:
    X, y = [], []
    for t in sorted({c.target for c in pool}):
        cands = _candidates(pool, t)
        act = _actual(pool, t)
        feats = _source_features(pool, t, cands)
        for a, b in itertools.combinations(cands, 2):
            if abs(act[a] - act[b]) < 1e-12:
                continue
            xa = np.array([feats[a][f] for f in FEATURES])
            xb = np.array([feats[b][f] for f in FEATURES])
            X.append(xa - xb)
            y.append(1.0 if act[a] > act[b] else 0.0)
    if not X:
        return np.zeros((0, len(FEATURES))), np.zeros(0)
    return np.array(X), np.array(y)


def _ranker_predictions(pool: Sequence[TransferCell], target: str,
                        candidates: Sequence[str],
                        w: Optional[np.ndarray] = None) -> Dict[str, float]:
    feats = _source_features(pool, target, candidates)
    if w is None:
        X, y = _pairwise_data(pool)
        w = _fit_pairwise(X, y) if len(y) else np.zeros(len(FEATURES))
    return {s: float(np.dot(w, [feats[s][f] for f in FEATURES])) for s in candidates}


def _best_fixed_source(pool: Sequence[TransferCell],
                       candidates: Sequence[str]) -> str:
    """The fixed source that maximises accuracy on the training pool."""
    pool_targets = sorted({c.target for c in pool})
    best, best_key = NONE, None
    for s in candidates:
        hits = 0
        for t in pool_targets:
            act = _actual(pool, t)
            cands = _candidates(pool, t)
            if s not in cands:
                continue
            if _choose(act, cands) == s:
                hits += 1
        key = (hits, -CHOICE_ORDER.index(s) if s in CHOICE_ORDER else -99)
        if best_key is None or key > best_key:
            best_key, best = key, s
    return best


# ---------------------------------------------------------------------------
# evaluation
# ---------------------------------------------------------------------------
def evaluate(cells: Sequence[TransferCell], mode: str,
             protocol: str = "leave_target") -> Dict[str, Any]:
    rows = []
    for t in _targets(cells):
        cands = _candidates(cells, t)
        pool = _train_pool(cells, t, protocol)
        if mode in ("cold", "global_mean", "source_mean"):
            preds = predict(pool, cands, mode)
        elif mode == "pairwise_rank":
            preds = _ranker_predictions(pool, t, cands)
        elif mode == "fixed_learned":
            s = _best_fixed_source(pool, cands)
            preds = {c: (1.0 if c == s else 0.0) for c in cands}
        elif mode.startswith("fixed_"):
            s = mode[len("fixed_"):]
            preds = {c: (1.0 if c == s else 0.0) for c in cands}
        elif mode == "val_rank":
            preds = {}
            for s in cands:
                if s == NONE:
                    preds[s] = 0.0
                    continue
                hits = [c.val_gain for c in cells if c.target == t and c.source == s]
                preds[s] = float(np.mean(hits)) if hits else 0.0
        else:
            raise ValueError(mode)

        chosen = _choose(preds, cands)
        act = _actual(cells, t)
        oracle = min(cands, key=lambda s: (-act[s], CHOICE_ORDER.index(s)))
        rows.append({
            "target": t, "kind": "champion" if len(cands) > 2 else "warm-start",
            "candidates": cands, "chosen": chosen, "oracle": oracle,
            "correct": bool(chosen == oracle),
            "chosen_gain": round(act[chosen], 6),
            "oracle_gain": round(act[oracle], 6),
            "regret": round(act[oracle] - act[chosen], 6),
        })

    multi = [r for r in rows if len(r["candidates"]) > 2]
    n = len(rows)
    return {
        "mode": mode,
        "n_targets": n,
        "n_multi_candidate": len(multi),
        "accuracy": round(sum(r["correct"] for r in rows) / n, 4) if n else 0.0,
        "accuracy_multi": (round(sum(r["correct"] for r in multi) / len(multi), 4)
                           if multi else 0.0),
        "avg_regret": round(sum(r["regret"] for r in rows) / n, 6) if n else 0.0,
        "avg_regret_multi": (round(sum(r["regret"] for r in multi) / len(multi), 6)
                             if multi else 0.0),
        "rows": rows,
    }


def build_report() -> Dict[str, Any]:
    cells = extract_cells(load_groups())
    modes = ["cold", "global_mean", "source_mean", "val_rank", "fixed_learned",
             "pairwise_rank"]
    for s in (NONE, "contract", "financial", "math"):
        modes.append(f"fixed_{s}")
    results = {m: evaluate(cells, m) for m in modes}

    # does a source-only ranker ever switch source across targets?
    switch = {}
    for m in ("pairwise_rank", "fixed_learned"):
        picks = {r["chosen"] for r in results[m]["rows"] if len(r["candidates"]) > 2}
        switch[m] = {"n_distinct_choices_multi": len(picks), "choices": sorted(picks)}

    # leave-one-target-out stability of the learned constant
    consts = []
    for t in _targets(cells):
        pool = _train_pool(cells, t, "leave_target")
        consts.append(_best_fixed_source(pool, _candidates(cells, t)))

    best_const = max(
        (m for m in modes if m.startswith("fixed_")),
        key=lambda m: (results[m]["accuracy_multi"], -results[m]["avg_regret_multi"]))
    return {
        "protocol": ("FUTURE_DIRECTIONS.md Priority 6: choose a transfer source by ranking "
                     "rather than by regression, trained leave-target-out"),
        "n_cells": len(cells), "n_targets": len(_targets(cells)),
        "features": list(FEATURES),
        "structural_limit": (
            "within a target every candidate shares the target-level features, so only "
            "source-level features can rank candidates; a source-only scorer is constant "
            "across targets and therefore equivalent to always choosing one fixed source"),
        "p6_target": {
            "criterion": "top-1 on the multi-candidate targets >= 2/3",
            "best_hindsight_constant": {
                "mode": best_const,
                "source": best_const[len("fixed_"):],
                "accuracy_multi": results[best_const]["accuracy_multi"],
                "avg_regret_multi": results[best_const]["avg_regret_multi"],
                "note": "chosen with knowledge of the labels, so it is a ceiling, not a result",
            },
            "best_learned": {
                "mode": "pairwise_rank",
                "accuracy_multi": results["pairwise_rank"]["accuracy_multi"],
                "avg_regret_multi": results["pairwise_rank"]["avg_regret_multi"],
            },
            "met_by_learned_ranker": (results["pairwise_rank"]["accuracy_multi"] >= 0.6666),
            "met_by_hindsight_constant": (results[best_const]["accuracy_multi"] >= 0.6666),
            "learned_fold_choices_unstable": len(set(consts)) > 1,
        },
        "results": results,
        "choice_variation": switch,
        "learned_constant_per_fold": consts,
    }


def _print(rep: Dict[str, Any]) -> None:
    print(f"cells={rep['n_cells']}  targets={rep['n_targets']}  "
          f"multi-candidate targets={rep['results']['cold']['n_multi_candidate']}")
    print(f"\n  {'mode':<18} {'acc':>6} {'acc_multi':>10} {'regret':>9} {'regret_multi':>13}")
    for m, r in rep["results"].items():
        print(f"  {m:<18} {r['accuracy']:>6.3f} {r['accuracy_multi']:>10.3f} "
              f"{r['avg_regret']:>9.6f} {r['avg_regret_multi']:>13.6f}")
    print("\n  multi-candidate picks:")
    for m in ("pairwise_rank", "fixed_learned", "source_mean", "val_rank"):
        picks = [f"{r['target'][:18]}->{r['chosen']}" for r in rep["results"][m]["rows"]
                 if len(r["candidates"]) > 2]
        print(f"    {m:<16} {'  '.join(picks)}")
    print(f"\n  learned constant per leave-target-out fold: {rep['learned_constant_per_fold']}")
    print(f"  distinct choices across multi targets: {rep['choice_variation']}")
    tg = rep["p6_target"]
    print(f"\n  P6 target ({tg['criterion']}):")
    print(f"    best hindsight constant   {tg['best_hindsight_constant']['source']:<9} "
          f"acc_multi={tg['best_hindsight_constant']['accuracy_multi']:.3f} "
          f"regret_multi={tg['best_hindsight_constant']['avg_regret_multi']:.6f}")
    print(f"    best learned ranker       pairwise  "
          f"acc_multi={tg['best_learned']['accuracy_multi']:.3f} "
          f"regret_multi={tg['best_learned']['avg_regret_multi']:.6f}")
    print(f"    met by learned ranker: {tg['met_by_learned_ranker']} | "
          f"met by hindsight constant: {tg['met_by_hindsight_constant']} | "
          f"fold choices unstable: {tg['learned_fold_choices_unstable']}")


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    args = ap.parse_args(argv)
    rep = build_report()
    _print(rep)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    path = out / "cross_task_rank_results.json"
    path.write_text(json.dumps(rep, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nwrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
