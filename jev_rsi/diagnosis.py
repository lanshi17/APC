"""Diagnostic analysis: why does val-argmax win?

Implements the three diagnostics proposed in ``FUTURE_DIRECTIONS.md``
(Priority 2) over the frozen corpus:

1. **Val-holdout alignment** -- Spearman rho between per-arm validation and
   holdout rankings, and whether rho separates the groups where val-argmax
   is right from the groups where it is wrong.
2. **Signal sparsity** -- how many arms are separated from the next-best by
   more than the noise band ("effective signal"), and whether structured
   gating ever helps where that count is high.
3. **OOD screening** -- can a simple pre-registered rule flag the groups
   where *no* val-based policy can succeed?  Rule selection is nested
   inside leave-one-group-out, so the reported accuracy is not the
   in-sample optimum.

The document's expected findings are recorded as hypotheses and are
reported as confirmed or refuted by measurement, not asserted.

Usage
-----
    python -m jev_rsi.diagnosis                     # full report
    python -m jev_rsi.diagnosis --out /tmp/x        # write elsewhere
"""
from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy.stats import rankdata, spearmanr

from .core import NOISE_BAND, Decision, JevGate, MetaParams, get_policy
from .data import Group, load_groups
from .experiments import summarize

DEFAULT_OUT = Path(__file__).resolve().parent / "results"


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _greedy_correct(g: Group) -> bool:
    return g.greedy.key == g.oracle.key


def _rho(g: Group) -> Tuple[float, float]:
    """Spearman rho(val, holdout) plus the share of arms tied on val.

    Ties matter here: many groups have several arms sitting at exactly
    ``val = 0.0`` (unscored cross-domain champions), so a raw Spearman
    coefficient is mostly driven by the tie block.
    """
    val = np.array([a.val for a in g.arms], dtype=float)
    hold = np.array([a.hold for a in g.arms], dtype=float)
    if len(val) < 2 or np.all(val == val[0]) or np.all(hold == hold[0]):
        rho = float("nan")
    else:
        rho = float(spearmanr(val, hold).statistic)
    _, counts = np.unique(val, return_counts=True)
    tie_share = float(max(counts) / len(val))
    return rho, tie_share


def _effective_signal(g: Group) -> int:
    """Arms separated from the next-best by more than the noise band."""
    scores = sorted((a.val for a in g.arms), reverse=True)
    effective = 1
    for i in range(1, len(scores)):
        if scores[i - 1] - scores[i] > NOISE_BAND:
            effective += 1
    return effective


def _features(g: Group) -> Dict[str, float]:
    val = np.array([a.val for a in g.arms], dtype=float)
    hold = np.array([a.hold for a in g.arms], dtype=float)
    srt = sorted(val, reverse=True)
    top2 = (srt[0] - srt[1]) if len(srt) > 1 else 0.0
    return {
        "val_variance": float(np.var(val)),
        "val_holdout_gap": float(val.max() - hold.max()),
        "top2_val_margin": float(top2),
        "n_arms": float(g.n_arms),
        "effective_signal": float(_effective_signal(g)),
        "tie_share": float(max(np.unique(val, return_counts=True)[1]) / len(val)),
        "is_transfer": float("transfer_" in g.gid),
        "is_champion": float("champ" in g.gid),
    }


def _auc(scores: Sequence[float], labels: Sequence[bool]) -> float:
    """Rank AUC of ``scores`` as a predictor of ``labels`` (ties = 0.5)."""
    pos = [s for s, l in zip(scores, labels) if l]
    neg = [s for s, l in zip(scores, labels) if not l]
    if not pos or not neg:
        return float("nan")
    r = rankdata(scores)
    ranks_pos = sum(r[i] for i, l in enumerate(labels) if l)
    return float((ranks_pos - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


# ---------------------------------------------------------------------------
# experiment 1 -- val/holdout rank alignment
# ---------------------------------------------------------------------------
def experiment_alignment(groups: Sequence[Group]) -> Dict[str, Any]:
    rows = []
    for g in groups:
        rho, tie_share = _rho(g)
        rows.append({"gid": g.gid, "task": g.task, "n_arms": g.n_arms,
                     "rho": None if np.isnan(rho) else round(rho, 4),
                     "tie_share": round(tie_share, 4),
                     "greedy_correct": _greedy_correct(g),
                     "greedy_regret": round(g.oracle.hold - g.greedy.hold, 6),
                     "is_informative": g.is_informative,
                     "informative": bool(g.is_informative)})

    valid = [r for r in rows if r["rho"] is not None]
    rhos = [r["rho"] for r in valid]
    labels = [r["greedy_correct"] for r in valid]

    buckets = {"rho >= 0.8": [], "0.3 <= rho < 0.8": [], "rho < 0.3": []}
    for r in valid:
        key = ("rho >= 0.8" if r["rho"] >= 0.8
               else "0.3 <= rho < 0.8" if r["rho"] >= 0.3 else "rho < 0.3")
        buckets[key].append(r)

    bucket_stats = {}
    for k, rs in buckets.items():
        if not rs:
            bucket_stats[k] = {"n": 0}
            continue
        bucket_stats[k] = {
            "n": len(rs),
            "greedy_correct": sum(1 for r in rs if r["greedy_correct"]),
            "greedy_accuracy": round(sum(1 for r in rs if r["greedy_correct"]) / len(rs), 4),
            "share_informative": round(sum(1 for r in rs if r["informative"]) / len(rs), 4),
        }

    zero_spread = [g.gid for g in groups if np.var([a.val for a in g.arms]) == 0]
    zero_info = [g.gid for g in groups
                 if np.var([a.val for a in g.arms]) == 0 and g.is_informative]
    return {
        "n_groups": len(groups),
        "n_scored": len(valid),
        "n_zero_val_spread": len(zero_spread),
        "zero_val_spread_gids": zero_spread,
        "n_zero_val_spread_unsolvable": len(zero_info),
        "zero_val_spread_unsolvable_gids": zero_info,
        "rho_summary": {
            "min": round(float(np.min(rhos)), 4),
            "median": round(float(np.median(rhos)), 4),
            "max": round(float(np.max(rhos)), 4),
            "n_negative": int(sum(1 for r in rhos if r < 0)),
            "n_below_0.3": int(sum(1 for r in rhos if r < 0.3)),
            "n_above_0.8": int(sum(1 for r in rhos if r >= 0.8)),
        },
        "buckets": bucket_stats,
        "auc_rho_predicts_greedy_correct": round(_auc(rhos, labels), 4),
        "hypothesis": "high rho -> greedy correct; low rho -> greedy fails (and so does Jev)",
        "rows": rows,
    }


# ---------------------------------------------------------------------------
# experiment 2 -- signal sparsity
# ---------------------------------------------------------------------------
def experiment_sparsity(groups: Sequence[Group]) -> Dict[str, Any]:
    gate = JevGate(meta=MetaParams())
    rows = []
    for g in groups:
        eff = _effective_signal(g)
        jev = gate.act(g)
        heur = get_policy("heuristic")(g)
        rows.append({
            "gid": g.gid, "n_arms": g.n_arms, "effective_signal": eff,
            "greedy_correct": _greedy_correct(g),
            "heuristic_correct": heur.regret < 1e-9,
            "jev_correct": jev.regret < 1e-9,
            "gate_wins_where_greedy_loses": (jev.regret < 1e-9) and not _greedy_correct(g),
            "is_informative": bool(g.is_informative),
        })

    by_eff: Dict[str, Any] = {}
    for eff in sorted({r["effective_signal"] for r in rows}):
        rs = [r for r in rows if r["effective_signal"] == eff]
        by_eff[str(eff)] = {
            "n": len(rs),
            "greedy_correct": sum(1 for r in rs if r["greedy_correct"]),
            "greedy_accuracy": round(sum(1 for r in rs if r["greedy_correct"]) / len(rs), 4),
            "jev_correct": sum(1 for r in rs if r["jev_correct"]),
            "gate_wins_where_greedy_loses": sum(1 for r in rs if r["gate_wins_where_greedy_loses"]),
        }

    eff_vals = [r["effective_signal"] for r in rows]
    return {
        "distribution": {str(k): int(v) for k, v in
                         zip(*np.unique(eff_vals, return_counts=True))} if rows else {},
        "by_effective_signal": by_eff,
        "max_effective_signal": int(max(eff_vals)) if eff_vals else 0,
        "n_groups_with_ge3": int(sum(1 for v in eff_vals if v >= 3)),
        "hypothesis": "effective 1-2 -> greedy wins; effective >= 3 -> structure has room",
        "rows": rows,
    }


# ---------------------------------------------------------------------------
# experiment 3 -- OOD screening rule (nested LOO)
# ---------------------------------------------------------------------------
def _candidate_rules(feature_names: Sequence[str]) -> List[Dict[str, Any]]:
    """Single-feature threshold rules and two-feature conjunctions."""
    rules: List[Dict[str, Any]] = []
    for f in feature_names:
        rules.append({"kind": "le", "features": [f], "ops": ["<="]})
        rules.append({"kind": "ge", "features": [f], "ops": [">="]})
    for f1, f2 in itertools.combinations(feature_names, 2):
        rules.append({"kind": "and", "features": [f1, f2], "ops": ["<=", "<="]})
        rules.append({"kind": "and", "features": [f1, f2], "ops": [">=", ">="]})
        rules.append({"kind": "and", "features": [f1, f2], "ops": ["<=", ">="]})
    return rules


def _fit_rule(rule: Dict[str, Any], X: np.ndarray, feats: Sequence[str],
              y: np.ndarray) -> Optional[Dict[str, Any]]:
    """Pick thresholds by balanced accuracy, then plain accuracy, then brevity.

    Balanced accuracy ties constantly on n=31 (many thresholds induce the same
    confusion matrix), so an explicit, deterministic tie-break is required:
    otherwise the fold-to-fold result is an artefact of list order.
    """
    idx = [feats.index(f) for f in rule["features"]]
    best: Optional[Tuple[Tuple[float, float, int], Dict[str, Any]]] = None
    grids = [np.unique(X[:, i]) for i in idx]
    for combo in itertools.product(*grids):
        thr = list(combo)
        pred = np.ones(len(y), dtype=bool)
        for i, t, op in zip(idx, thr, rule["ops"]):
            pred &= (X[:, i] <= t) if op == "<=" else (X[:, i] >= t)
        tp = int(np.sum(pred & y)); fn = int(np.sum(~pred & y))
        fp = int(np.sum(pred & ~y)); tn = int(np.sum(~pred & ~y))
        if tp + fn == 0 or tn + fp == 0:
            continue
        bal = 0.5 * (tp / (tp + fn) + tn / (tn + fp))
        acc = (tp + tn) / len(y)
        cand = {"balanced_accuracy": round(float(bal), 4), "accuracy": round(float(acc), 4),
                "thresholds": [float(t) for t in thr],
                "tp": tp, "fp": fp, "fn": fn, "tn": tn,
                "n_terms": len(rule["features"])}
        key = (cand["balanced_accuracy"], cand["accuracy"], -cand["n_terms"])
        if best is None or key > best[0]:
            best = (key, cand)
    return None if best is None else best[1]


def _apply_rule(rule: Dict[str, Any], fit: Dict[str, Any], row: np.ndarray,
                feats: Sequence[str]) -> bool:
    pred = True
    for f, t, op in zip(rule["features"], fit["thresholds"], rule["ops"]):
        v = row[feats.index(f)]
        pred &= (v <= t) if op == "<=" else (v >= t)
    return bool(pred)


def _confusion(pred: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
    tp = int(np.sum(pred & y)); fn = int(np.sum(~pred & y))
    fp = int(np.sum(pred & ~y)); tn = int(np.sum(~pred & ~y))
    sens = tp / (tp + fn) if tp + fn else 0.0
    spec = tn / (tn + fp) if tn + fp else 0.0
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "sensitivity": round(sens, 4), "specificity": round(spec, 4),
            "balanced_accuracy": round(0.5 * (sens + spec), 4),
            "accuracy": round((tp + tn) / len(y), 4) if len(y) else 0.0}


def experiment_ood(groups: Sequence[Group]) -> Dict[str, Any]:
    feats = ["val_variance", "val_holdout_gap", "top2_val_margin", "n_arms",
             "effective_signal", "tie_share", "is_transfer", "is_champion"]
    X = np.array([[float(_features(g)[f]) for f in feats] for g in groups])
    # "unsolvable by val" == val-argmax is wrong by more than the noise band
    y = np.array([g.is_informative for g in groups], dtype=bool)

    rows = []
    for g, yi in zip(groups, y):
        f = _features(g)
        rows.append({"gid": g.gid, **{k: round(v, 6) for k, v in f.items()},
                     "unsolvable_by_val": bool(yi)})

    # (a) unfitted, interpretable rules -- no selection, so no optimistic bias
    unfitted = {
        "all arms tie on val (var == 0)": np.array([np.var([a.val for a in g.arms]) == 0 for g in groups], bool),
        "any arm has val == 0": np.array([any(a.val == 0 for a in g.arms) for g in groups], bool),
        "n_arms >= 6": np.array([g.n_arms >= 6 for g in groups], bool),
        "effective_signal <= 1": np.array([_effective_signal(g) <= 1 for g in groups], bool),
    }
    unfitted_stats = {name: _confusion(pred, y) for name, pred in unfitted.items()}

    # (b) nested leave-one-group-out selection
    rules = _candidate_rules(feats)
    picks: List[str] = []
    feature_freq: Dict[str, int] = {f: 0 for f in feats}
    loo_pred = np.zeros(len(y), dtype=bool)
    for k in range(len(y)):
        mask = np.ones(len(y), dtype=bool); mask[k] = False
        best = None
        for r in rules:
            fit = _fit_rule(r, X[mask], feats, y[mask])
            if fit is None:
                continue
            key = (fit["balanced_accuracy"], fit["accuracy"], -fit["n_terms"])
            if best is None or key > best[0]:
                best = (key, r, fit)
        if best is None:
            continue
        _, r, fit = best
        picks.append(" AND ".join(f"{f} {o} {round(t, 4)}"
                                  for f, o, t in zip(r["features"], r["ops"], fit["thresholds"])))
        for f in r["features"]:
            feature_freq[f] += 1
        loo_pred[k] = _apply_rule(r, fit, X[k], feats)

    # (c) in-sample optimum, for the optimism gap
    in_sample_best, in_sample_fit = None, None
    for r in rules:
        fit = _fit_rule(r, X, feats, y)
        if fit and (in_sample_fit is None
                    or (fit["balanced_accuracy"], fit["accuracy"]) > (in_sample_fit["balanced_accuracy"], in_sample_fit["accuracy"])):
            in_sample_best, in_sample_fit = r, fit

    loo_conf = _confusion(loo_pred, y)
    return {
        "n_groups": len(groups),
        "n_unsolvable_by_val": int(y.sum()),
        "majority_baseline_accuracy": round(max(float(y.mean()), 1 - float(y.mean())), 4),
        "unfitted_rules": unfitted_stats,
        "nested_loo": {**loo_conf,
                       "n_distinct_rules_picked": len(set(picks)),
                       "feature_frequency": feature_freq,
                       "picks": picks},
        "in_sample_best_rule": None if in_sample_best is None else {
            "description": " AND ".join(f"{f} {o} {round(t, 4)}"
                                        for f, o, t in zip(in_sample_best["features"],
                                                           in_sample_best["ops"],
                                                           in_sample_fit["thresholds"])),
            **in_sample_fit},
        "features": feats,
        "hypothesis": "an identifiable OOD rule flags the groups where val cannot work",
        "rows": rows,
    }


# ---------------------------------------------------------------------------
# report
# ---------------------------------------------------------------------------
def build_report() -> Dict[str, Any]:
    groups = load_groups()
    decision = [g for g in groups if g.is_decision]
    report = {
        "protocol": "FUTURE_DIRECTIONS.md Priority 2 diagnostics over the frozen corpus",
        "n_groups_total": len(groups),
        "n_decision_groups": len(decision),
        "n_informative": sum(1 for g in decision if g.is_informative),
        "noise_band": NOISE_BAND,
        "experiment_1_alignment": experiment_alignment(decision),
        "experiment_2_sparsity": experiment_sparsity(decision),
        "experiment_3_ood": experiment_ood(decision),
    }
    return report


def _print(report: Dict[str, Any]) -> None:
    e1 = report["experiment_1_alignment"]
    print("=== E1: val/holdout rank alignment ===")
    print(f"  scored groups: {e1['n_scored']}/{e1['n_groups']}   rho: {e1['rho_summary']}")
    print(f"  groups with zero val spread (all arms val=0): {e1['n_zero_val_spread']}"
          f"  of which unsolvable-by-val: {e1['n_zero_val_spread_unsolvable']}")
    for k, v in e1["buckets"].items():
        if v.get("n"):
            print(f"    {k:<18} n={v['n']:<3} greedy-correct {v['greedy_correct']}/{v['n']} "
                  f"({v['greedy_accuracy']:.0%})  informative-share {v['share_informative']:.0%}")
    print(f"  AUC(rho -> greedy correct) = {e1['auc_rho_predicts_greedy_correct']}")

    e2 = report["experiment_2_sparsity"]
    print("\n=== E2: signal sparsity ===")
    print(f"  distribution: {e2['distribution']}   max={e2['max_effective_signal']}")
    for k, v in e2["by_effective_signal"].items():
        print(f"    effective={k:<3} n={v['n']:<3} greedy {v['greedy_correct']}/{v['n']} "
              f"jev {v['jev_correct']}/{v['n']}  gate-wins-where-greedy-loses {v['gate_wins_where_greedy_loses']}")

    e3 = report["experiment_3_ood"]
    print("\n=== E3: OOD screening ===")
    print(f"  unsolvable-by-val groups: {e3['n_unsolvable_by_val']}/{e3['n_groups']}"
          f"   majority baseline accuracy {e3['majority_baseline_accuracy']:.4f}")
    print("  unfitted (no selection, no optimism):")
    for name, c in e3["unfitted_rules"].items():
        print(f"    {name:<34} bal-acc {c['balanced_accuracy']:.3f}  acc {c['accuracy']:.4f}"
              f"  sens {c['sensitivity']:.3f}  spec {c['specificity']:.3f}")
    nl = e3["nested_loo"]
    print(f"  nested-LOO selection: bal-acc {nl['balanced_accuracy']:.3f}  acc {nl['accuracy']:.4f}"
          f"   ({nl['n_distinct_rules_picked']} distinct rules across folds)")
    top = sorted(nl["feature_frequency"].items(), key=lambda kv: -kv[1])[:4]
    print(f"    most-picked features: {top}")
    if e3["in_sample_best_rule"]:
        r = e3["in_sample_best_rule"]
        print(f"  in-sample best (optimistic): {r['description']}  bal-acc {r['balanced_accuracy']}")


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    args = ap.parse_args(argv)

    report = build_report()
    _print(report)

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    path = out / "diagnosis_results.json"
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nwrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
