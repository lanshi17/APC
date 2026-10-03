"""Experiment runners for the Jev + RSI study.

Every number produced here is measured on the frozen APC artefact
(``experiments/apc_full_dataset.json``).  No result is written by hand.

Protocols
---------
* ``full``        -- offline replay over all scorable groups (train == test,
                     i.e. the RSI variant is *not* used here).
* ``loo``         -- leave-one-group-out: for each decision group, RSI is trained
                     on every other decision group and then evaluated on the
                     held-out group.  Training and test groups are disjoint.
* ``curve``       -- RSI trained on the first k groups of a fixed order,
                     evaluated on the remaining groups.
* ``ablation``    -- component removal + mechanism replacement, under ``loo``.
* ``robustness``  -- val-score noise injection + meta-parameter grid.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .core import (N_MC, NOISE_BAND, Decision, JevGate, MetaParams, get_policy)
from .data import Arm, Group, load_groups
from .rsi import RSILearner, replay
from . import cross_task as cross_task_mod

RESULTS = Path(__file__).resolve().parent / "results"


# ---------------------------------------------------------------------------
# Harness
# ---------------------------------------------------------------------------
class Harness:
    def __init__(self, groups: Optional[List[Group]] = None):
        self.groups = groups if groups is not None else load_groups()
        self.feats: Dict[str, Dict[str, float]] = {
            g.gid: JevGate.features(g) for g in self.groups
        }

    # -- evaluation -------------------------------------------------------
    def evaluate(self, policy_name: str, evaluation: Sequence[Group],
                 meta: Optional[MetaParams] = None,
                 enabled: Sequence[str] = JevGate.ALL,
                 fallback: str = "base") -> List[Decision]:
        if policy_name in ("conservative", "greedy", "heuristic"):
            fn = get_policy(policy_name)
            return [fn(g) for g in evaluation]
        gate = JevGate(meta=meta, enabled=enabled, fallback=fallback)
        return [gate.act(g, self.feats[g.gid]) for g in evaluation]

    @property
    def decision_groups(self) -> List[Group]:
        return [g for g in self.groups if g.is_decision]

    @property
    def informative_groups(self) -> List[Group]:
        return [g for g in self.groups if g.is_informative]

    @property
    def collapse_groups(self) -> List[Group]:
        return [g for g in self.groups if g.has_collapse and g.is_decision]


def summarize(groups: Sequence[Group], decisions: Sequence[Decision]) -> dict:
    regs = [d.regret for d in decisions]
    n = len(regs)
    exact = sum(1 for r in regs if r < 1e-9)
    wn = sum(1 for r in regs if r < NOISE_BAND)
    collapse = [d for g, d in zip(groups, decisions) if g.has_collapse and g.is_decision]
    return {
        "n_groups": n,
        "exact_oracle": exact,
        "exact_rate": round(exact / n, 4) if n else 0.0,
        "within_noise": wn,
        "within_noise_rate": round(wn / n, 4) if n else 0.0,
        "avg_regret": round(float(np.mean(regs)), 6) if n else 0.0,
        "max_regret": round(float(np.max(regs)), 6) if n else 0.0,
        "total_regret": round(float(np.sum(regs)), 6) if n else 0.0,
        "n_collapse_cases": len(collapse),
        "collapse_avoided": sum(1 for d in collapse if d.regret < NOISE_BAND),
    }


def seen(s: dict) -> str:
    return (f"{s['exact_oracle']:>2}/{s['n_groups']:<2} oracle  "
            f"{s['within_noise']:>2}/{s['n_groups']:<2} within-noise  "
            f"avg-regret {s['avg_regret']:+.5f}")


# ---------------------------------------------------------------------------
# 1. Main table
# ---------------------------------------------------------------------------
def main_table(h: Harness) -> dict:
    out: dict = {"dataset": dataset_audit(h), "policies": {}}
    sets = {"all": h.groups, "decision": h.decision_groups,
            "informative": h.informative_groups}
    for pname in ("conservative", "greedy", "heuristic"):
        out["policies"][pname] = {
            k: summarize(gs, h.evaluate(pname, gs)) for k, gs in sets.items()
        }
    out["policies"]["jev_static"] = {
        k: summarize(gs, h.evaluate("jev", gs)) for k, gs in sets.items()
    }
    return out


def dataset_audit(h: Harness) -> dict:
    return {
        "n_groups": len(h.groups),
        "n_arms": sum(g.n_arms for g in h.groups),
        "n_decision_groups": len(h.decision_groups),
        "n_informative_groups": len(h.informative_groups),
        "n_collapse_groups": len(h.collapse_groups),
        "greedy_total_regret": round(
            sum(g.oracle.hold - g.greedy.hold for g in h.groups), 6),
        "oracle_headroom_groups": [g.gid for g in h.informative_groups],
        "collapse_group_ids": [g.gid for g in h.collapse_groups],
    }


# ---------------------------------------------------------------------------
# 2. Leave-one-out RSI
# ---------------------------------------------------------------------------
def loo_rsi(h: Harness, clustering: bool = True,
            enabled: Sequence[str] = JevGate.ALL,
            meta0: Optional[MetaParams] = None,
            fallback: str = "base") -> dict:
    groups = h.decision_groups
    rows = []
    static_gate = JevGate(meta=meta0, enabled=enabled, fallback=fallback)
    for i, test in enumerate(groups):
        train = [g for j, g in enumerate(groups) if j != i]
        learner = replay(train, clustering=clustering, meta0=meta0,
                         enabled=enabled, fallback=fallback)
        gate = JevGate(meta=learner.meta, enabled=enabled, fallback=fallback)
        d_rsi = gate.act(test, h.feats[test.gid])
        d_static = static_gate.act(test, h.feats[test.gid])
        rows.append({
            "gid": test.gid,
            "family": test.family,
            "oracle": test.oracle.name,
            "rsi_deployed": d_rsi.deployed_name,
            "rsi_reason": d_rsi.reason,
            "rsi_regret": round(d_rsi.regret, 6),
            "static_deployed": d_static.deployed_name,
            "static_reason": d_static.reason,
            "static_regret": round(d_static.regret, 6),
            "learned_meta": learner.meta.as_dict(),
            "improvement": round(d_static.regret - d_rsi.regret, 6),
        })
    decisions = [Decision(r["rsi_deployed"], r["rsi_deployed"], r["rsi_reason"],
                          hold=0.0, regret=r["rsi_regret"]) for r in rows]
    static = [Decision(r["static_deployed"], r["static_deployed"],
                       r["static_reason"], hold=0.0, regret=r["static_regret"])
              for r in rows]
    metas = np.array([[r["learned_meta"][k] for k in
                       ("collapse_threshold", "alignment_threshold",
                        "confidence_floor")] for r in rows])
    return {
        "clustering": clustering,
        "enabled": list(enabled),
        "fallback": fallback,
        "rows": rows,
        "rsi": summarize(groups, decisions),
        "jev_static": summarize(groups, static),
        "learned_meta_mean": {
            k: round(float(v), 4) for k, v in
            zip(("collapse_threshold", "alignment_threshold", "confidence_floor"),
                metas.mean(axis=0))
        },
        "learned_meta_std": {
            k: round(float(v), 4) for k, v in
            zip(("collapse_threshold", "alignment_threshold", "confidence_floor"),
                metas.std(axis=0))
        },
        "wins": sum(1 for r in rows if r["improvement"] > 1e-9),
        "losses": sum(1 for r in rows if r["improvement"] < -1e-9),
        "ties": sum(1 for r in rows if abs(r["improvement"]) <= 1e-9),
    }


# ---------------------------------------------------------------------------
# 3. Learning curve
# ---------------------------------------------------------------------------
def learning_curve(h: Harness) -> dict:
    groups = sorted(h.decision_groups, key=lambda g: g.gid)
    order = [g.gid for g in groups]
    ks = [0, 3, 5, 8, 12, 16, 20, len(groups) - 5]
    ks = sorted({max(0, min(k, len(groups) - 1)) for k in ks})
    curve = []
    for k in ks:
        train, test = groups[:k], groups[k:]
        meta0 = MetaParams()
        if k == 0:
            meta = meta0
            learner_meta = meta0.as_dict()
        else:
            learner = replay(train, clustering=True, meta0=meta0)
            meta = learner.meta
            learner_meta = meta.as_dict()
        gate = JevGate(meta=meta)
        ds = [gate.act(g, h.feats[g.gid]) for g in test]
        s = summarize(test, ds)
        s["k_train"] = k
        s["learned_meta"] = learner_meta
        curve.append(s)
    return {"order": order, "points": curve}


# ---------------------------------------------------------------------------
# 4. Ablations
# ---------------------------------------------------------------------------
def ablations(h: Harness) -> dict:
    out: dict = {}
    out["component"] = {
        "full": loo_rsi(h, clustering=True),
        "no_rsi": {"note": "Jev gate with frozen default meta-parameters",
                   "summary": summarize(h.decision_groups,
                                        h.evaluate("jev", h.decision_groups))},
        "no_cluster": loo_rsi(h, clustering=False),
        "no_noul": loo_rsi(h, clustering=True,
                           enabled=("alignment", "confidence")),
        "no_score": loo_rsi(h, clustering=True,
                            enabled=("collapse", "confidence")),
        "no_confidence": loo_rsi(h, clustering=True,
                                 enabled=("collapse", "alignment")),
    }
    out["mechanism"] = mechanism_ablations(h)
    return out


def mechanism_ablations(h: Harness) -> dict:
    groups = h.decision_groups
    res: dict = {}

    # (a) single threshold rule on the worst arm
    ds = []
    for g in groups:
        worst = min(g.arms, key=lambda a: a.val)
        if worst.val < 0.2:
            arm = g.conservative
        else:
            arm = g.greedy
        ds.append(_mk(g, arm, "threshold_rule"))
    res["threshold_rule"] = summarize(groups, ds)

    # (b) logistic regression on gate features predicting regret, LOO-trained
    X = np.array([[h.feats[g.gid][k] for k in
                   ("margin", "collapse_noul", "alignment", "confidence",
                    "n_arms")] for g in groups])
    y = np.array([1.0 if g.oracle.hold - g.greedy.hold >= NOISE_BAND else 0.0
                  for g in groups])
    ds = []
    for i, g in enumerate(groups):
        mask = np.ones(len(groups), bool); mask[i] = False
        w, mu, sd = _logreg(X[mask], y[mask])
        p_bad = _sigmoid(np.append((X[i] - mu) / sd, 1.0) @ w)
        arm = g.conservative if p_bad > 0.5 else g.greedy
        ds.append(_mk(g, arm, f"logreg_p={p_bad:.3f}"))
    res["logistic_regression"] = summarize(groups, ds)
    res["logistic_regression"]["positive_labels"] = int(y.sum())

    # (c) Bayesian-only: deploy the conservative arm whenever the posterior
    #     P(best) of the val-argmax is below 0.5, else the val-argmax.
    ds = []
    for g in groups:
        f = h.feats[g.gid]
        arm = g.conservative if f["confidence"] < 0.5 else g.greedy
        ds.append(_mk(g, arm, "bayesian_only"))
    res["bayesian_only"] = summarize(groups, ds)

    # (d) static Jev gate for reference
    res["jev_static"] = summarize(groups, h.evaluate("jev", groups))
    res["heuristic"] = summarize(groups, h.evaluate("heuristic", groups))
    return res


def _mk(g: Group, arm: Arm, reason: str) -> Decision:
    return Decision(arm.key, arm.name, reason, arm.hold,
                    max(a.hold for a in g.arms) - arm.hold, {})


def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))


def _logreg(X: np.ndarray, y: np.ndarray, lr: float = 0.1,
            iters: int = 800):
    mu, sd = X.mean(axis=0), X.std(axis=0) + 1e-9
    Xs = np.hstack([(X - mu) / sd, np.ones((len(X), 1))])
    w = np.zeros(Xs.shape[1])
    for _ in range(iters):
        p = _sigmoid(Xs @ w)
        w -= lr * (Xs.T @ (p - y) / len(y) + 1e-3 * w)
    return w, mu, sd


# ---------------------------------------------------------------------------
# 5. Robustness
# ---------------------------------------------------------------------------
def robustness(h: Harness) -> dict:
    out: dict = {"noise": {}, "sweep": {}}
    rng = np.random.default_rng(4242)

    for level in (0.0, 0.02, 0.05, 0.10):
        noisy_groups = _perturb(h.groups, rng, level, mode="val")
        out["noise"][f"val_{level:.2f}"] = _evaluate_on(noisy_groups)

    for frac in (0.0, 0.25):
        dropped = _drop_arms(h.groups, rng, frac)
        out["noise"][f"drop_{frac:.2f}"] = _evaluate_on(dropped)
    out["note"] = ("noise/drop rows use the frozen static Jev gate so that the "
                   "input perturbation is the only varying factor; `sweep` "
                   "covers the meta-parameter response surface.")

    grid = []
    for ct in (0.4, 0.5, 0.6, 0.7, 0.8):
        for at in (0.2, 0.35, 0.5, 0.65):
            for cf in (0.4, 0.5, 0.6, 0.7):
                meta = MetaParams(ct, at, cf)
                s = summarize(h.decision_groups,
                              h.evaluate("jev", h.decision_groups, meta=meta))
                grid.append({"meta": meta.as_dict(), **s})
    grid.sort(key=lambda r: (-r["exact_oracle"], -r["within_noise"],
                             r["avg_regret"]))
    out["sweep"] = {"n_configs": len(grid), "top": grid[:12],
                    "worst": grid[-5:], "grid": grid}
    return out


def _evaluate_on(groups: List[Group]) -> dict:
    """Evaluate on perturbed copies of the groups (features recomputed)."""
    ds = []
    for g in groups:
        f = JevGate.features(g)
        gate = JevGate()
        ds.append(gate.act(g, f))
    return summarize(groups, ds)


# ---------------------------------------------------------------------------
# 6. Deep dive: rule usage, paired tests, cross-model split, oracle thresholds
# ---------------------------------------------------------------------------
GRID = {
    "collapse_threshold": (0.2, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9),
    "alignment_threshold": (0.05, 0.15, 0.3, 0.5, 0.7),
    "confidence_floor": (0.05, 0.2, 0.4, 0.6, 0.8),
}


def deep_dive(h: Harness) -> dict:
    groups = h.decision_groups
    out: dict = {}

    # (a) which rule actually fires?
    def reason_dist(meta=None):
        gate = JevGate(meta=meta)
        c: Dict[str, int] = {}
        for g in groups:
            r = gate.act(g, h.feats[g.gid]).reason
            c[r] = c.get(r, 0) + 1
        return c

    out["reason_distribution_default"] = reason_dist()
    out["reason_distribution_sweep_best"] = reason_dist(MetaParams(0.8, 0.2, 0.4))

    # (b) head-to-head tests on the decision groups
    g_dec = [d.regret for d in h.evaluate("greedy", groups)]
    h_dec = [d.regret for d in h.evaluate("heuristic", groups)]
    j_dec = [d.regret for d in h.evaluate("jev", groups)]
    out["paired_tests"] = {
        "greedy_vs_heuristic": _paired(g_dec, h_dec, "greedy", "heuristic"),
        "greedy_vs_jev_static": _paired(g_dec, j_dec, "greedy", "jev_static"),
    }

    # (c) cross-model split: does the gate transfer across the model variant
    #     that produced the arm scores?
    for label, pred in (("gpt6_variants", lambda g: "gpt6" in g.gid),
                        ("qwen_base", lambda g: "gpt6" not in g.gid)):
        sub = [g for g in groups if pred(g)]
        if not sub:
            continue
        out.setdefault("cross_model", {})[label] = {
            "n": len(sub),
            "greedy": summarize(sub, h.evaluate("greedy", sub)),
            "heuristic": summarize(sub, h.evaluate("heuristic", sub)),
            "jev_static": summarize(sub, h.evaluate("jev", sub)),
        }

    # (d) strongest possible threshold learner: grid search on the training
    #     folds, evaluated on the held-out group (leave-one-out).
    rows = []
    for i, test in enumerate(groups):
        train = [g for j, g in enumerate(groups) if j != i]
        best, best_key = None, None
        for ct in GRID["collapse_threshold"]:
            for at in GRID["alignment_threshold"]:
                for cf in GRID["confidence_floor"]:
                    meta = MetaParams(ct, at, cf)
                    s = summarize(train, h.evaluate("jev", train, meta=meta))
                    key = (s["exact_oracle"], s["within_noise"], -s["avg_regret"])
                    if best_key is None or key > best_key:
                        best_key, best = key, meta
        d = JevGate(meta=best).act(test, h.feats[test.gid])
        s = summarize(train, h.evaluate("jev", train, meta=best))
        rows.append({"gid": test.gid, "meta": best.as_dict(),
                     "train_exact": s["exact_oracle"],
                     "test_regret": round(d.regret, 6)})
    ds = [Decision(r["gid"], r["gid"], "oracle_threshold", hold=0.0,
                   regret=r["test_regret"]) for r in rows]
    out["oracle_threshold_loo"] = {**summarize(groups, ds),
                                   "rows": rows,
                                   "grid_size": len(GRID["collapse_threshold"]) *
                                   len(GRID["alignment_threshold"]) *
                                   len(GRID["confidence_floor"])}

    # (e) in-sample upper bound of the same grid (over-fitted reference point)
    best, best_key = None, None
    for ct in GRID["collapse_threshold"]:
        for at in GRID["alignment_threshold"]:
            for cf in GRID["confidence_floor"]:
                meta = MetaParams(ct, at, cf)
                s = summarize(groups, h.evaluate("jev", groups, meta=meta))
                key = (s["exact_oracle"], s["within_noise"], -s["avg_regret"])
                if best_key is None or key > best_key:
                    best_key, best = key, meta
    out["grid_insample_best"] = {
        "meta": best.as_dict(),
        **summarize(groups, h.evaluate("jev", groups, meta=best)),
    }

    # (f) sensitivity to the fallback-arm definition (design decision made
    #     before results were observed, reported for full transparency)
    out["fallback_sensitivity"] = {
        "base_static": summarize(groups, h.evaluate("jev", groups, fallback="base")),
        "occam_static": summarize(groups, h.evaluate("jev", groups, fallback="occam")),
        "base_loo_rsi": loo_rsi(h, fallback="base")["rsi"],
        "occam_loo_rsi": loo_rsi(h, fallback="occam")["rsi"],
    }
    return out


# ---------------------------------------------------------------------------
# 6b. Data-quality sensitivity: the corpus contains cells that the APC incident
#     analysis explicitly quarantined (gateway service drift).  Re-report the
#     headline comparison without them.
# ---------------------------------------------------------------------------
def data_quality_sensitivity(h: Harness) -> dict:
    clean = [g for g in h.decision_groups
             if "quarantine" not in g.gid and "servicedrift" not in g.gid]
    out = {
        "n_decision_all": len(h.decision_groups),
        "n_decision_clean": len(clean),
        "excluded": [g.gid for g in h.decision_groups if g not in clean],
        "clean": {
            "greedy": summarize(clean, h.evaluate("greedy", clean)),
            "heuristic": summarize(clean, h.evaluate("heuristic", clean)),
            "jev_static": summarize(clean, h.evaluate("jev", clean)),
        },
    }
    # LOO-RSI restricted to the clean groups
    rows = []
    for i, test in enumerate(clean):
        train = [g for j, g in enumerate(clean) if j != i]
        learner = replay(train, clustering=True)
        d = JevGate(meta=learner.meta).act(test, h.feats[test.gid])
        rows.append(d)
    out["clean"]["loo_rsi"] = summarize(clean, rows)
    out["clean"]["paired_greedy_vs_jev"] = _paired(
        [d.regret for d in h.evaluate("greedy", clean)],
        [d.regret for d in h.evaluate("jev", clean)], "greedy", "jev_static")
    return out


# ---------------------------------------------------------------------------
# 7. Reproduction check against the frozen 6-group artefact
# ---------------------------------------------------------------------------
HIST_MAP = {
    "financial-s43": "financial-s43",
    "financial-s44": "financial-s44",
    "math-s42": "math-s42",
    "transfer_contract_to_math": "transfer_contract_to_math-s42",
    "transfer_financial_to_math_gpt6": "transfer_financial_to_math_gpt6-s42",
    "transfer_math_to_contract": "transfer_math_to_contract-s42",
}


def historical_reproduction(h: Harness) -> dict:
    import json as _json
    from .data import APC_ROOT
    art_path = APC_ROOT / "experiments" / "jev_vs_heuristic_comparison.json"
    art = _json.loads(art_path.read_text())
    by = {g.gid: g for g in h.groups}
    groups = [by[HIST_MAP[r["group"]]] for r in art["results"]
              if HIST_MAP.get(r["group"]) in by]
    return {
        "source_artifact": str(art_path),
        "frozen_artifact": art["summary"],
        "replayed": {
            "heuristic": summarize(groups, h.evaluate("heuristic", groups)),
            "greedy": summarize(groups, h.evaluate("greedy", groups)),
            "jev_static": summarize(groups, h.evaluate("jev", groups)),
        },
        "n_groups_replayed": len(groups),
        "note": ("rule *labels* differ from the frozen artefact (our alignment "
                 "primitive is margin-based, the artefact's was sample-size "
                 "based) but the deployed arms and regrets agree."),
    }


def _paired(a: Sequence[float], b: Sequence[float], na: str, nb: str) -> dict:
    """Bootstrap CI + sign test for paired regret differences a - b."""
    from scipy.stats import binomtest
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    diff = a - b
    rng = np.random.default_rng(20261002)
    idx = rng.integers(0, len(diff), size=(10000, len(diff)))
    boot = diff[idx].mean(axis=1)
    wins = int((diff > 1e-9).sum())
    losses = int((diff < -1e-9).sum())
    p = binomtest(wins, wins + losses, 0.5).pvalue if (wins + losses) else 1.0
    return {
        "pair": f"{na} - {nb}",
        "mean_diff": round(float(diff.mean()), 6),
        "ci95": [round(float(np.percentile(boot, 2.5)), 6),
                 round(float(np.percentile(boot, 97.5)), 6)],
        "wins": wins, "losses": losses,
        "ties": int(len(diff) - wins - losses),
        "sign_test_p": round(float(p), 4),
        "interpretation": ("negative mean_diff means lower regret for " + na),
    }


def _perturb(groups: List[Group], rng, level: float, mode: str = "val") -> List[Group]:
    out = []
    for g in groups:
        arms = [Arm(key=a.key, name=a.name,
                    val=a.val + float(rng.normal(0, level)) if level else a.val,
                    hold=a.hold, rank=a.rank, simplicity=a.simplicity)
                for a in g.arms]
        out.append(Group(g.gid, g.task, g.family, arms))
    return out


def _drop_arms(groups: List[Group], rng, frac: float) -> List[Group]:
    out = []
    for g in groups:
        arms = list(g.arms)
        if frac > 0 and len(arms) > 2:
            n_drop = max(1, int(round(frac * len(arms))))
            idx = rng.choice(len(arms), size=n_drop, replace=False)
            arms = [a for i, a in enumerate(arms) if i not in set(idx.tolist())]
        out.append(Group(g.gid, g.task, g.family, arms))
    return out


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def run_all(outdir: Path = RESULTS) -> dict:
    outdir.mkdir(parents=True, exist_ok=True)
    h = Harness()
    payload = {
        "main_table": main_table(h),
        "loo_rsi": loo_rsi(h),
        "learning_curve": learning_curve(h),
        "ablations": ablations(h),
        "robustness": robustness(h),
        "deep_dive": deep_dive(h),
        "data_quality": data_quality_sensitivity(h),
        "historical_reproduction": historical_reproduction(h),
        "cross_task": cross_task_mod.run(h.groups),
    }
    (outdir / "jev_rsi_results.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False))
    return payload


def _fmt(payload: dict) -> str:
    L = []
    a = payload["main_table"]["dataset"]
    L.append("=== DATASET ===")
    L.append(f"groups={a['n_groups']} arms={a['n_arms']} "
             f"decision={a['n_decision_groups']} informative={a['n_informative_groups']} "
             f"collapse={a['n_collapse_groups']} greedy_regret={a['greedy_total_regret']}")
    L.append("")
    L.append("=== MAIN TABLE ===")
    for p, sets in payload["main_table"]["policies"].items():
        L.append(f"{p:14s} all: {seen(sets['all'])}")
        L.append(f"{'':14s} dec: {seen(sets['decision'])}")
        L.append(f"{'':14s} inf: {seen(sets['informative'])}")
    loo = payload["loo_rsi"]
    L.append("")
    L.append("=== LOO RSI ===")
    L.append(f"RSI        {seen(loo['rsi'])}  wins={loo['wins']} losses={loo['losses']} ties={loo['ties']}")
    L.append(f"JevStatic  {seen(loo['jev_static'])}")
    L.append(f"learned meta mean={loo['learned_meta_mean']} std={loo['learned_meta_std']}")
    L.append("")
    L.append("=== LEARNING CURVE ===")
    for p in payload["learning_curve"]["points"]:
        L.append(f"k={p['k_train']:<3d} {seen(p)}  meta={p['learned_meta']}")
    L.append("")
    L.append("=== ABLATION (component) ===")
    c = payload["ablations"]["component"]
    for k in ("full", "no_rsi", "no_cluster", "no_noul", "no_score", "no_confidence"):
        s = c[k].get("rsi") or c[k].get("summary")
        L.append(f"{k:14s} {seen(s)}")
    L.append("")
    L.append("=== ABLATION (mechanism) ===")
    for k, s in payload["ablations"]["mechanism"].items():
        L.append(f"{k:22s} {seen(s)}")
    L.append("")
    L.append("=== ROBUSTNESS ===")
    for k, s in payload["robustness"]["noise"].items():
        L.append(f"{k:12s} {seen(s)}")
    top = payload["robustness"]["sweep"]["top"][0]
    L.append(f"sweep best {top['meta']} -> {seen(top)}")
    L.append("")
    L.append("=== DEEP DIVE ===")
    dd = payload["deep_dive"]
    L.append(f"rule fires (default)   : {dd['reason_distribution_default']}")
    L.append(f"rule fires (sweep best): {dd['reason_distribution_sweep_best']}")
    for k, v in dd["paired_tests"].items():
        L.append(f"{k:24s} {v['pair']} mean_diff={v['mean_diff']:+.5f} "
                 f"CI={v['ci95']} W/L/T={v['wins']}/{v['losses']}/{v['ties']} "
                 f"p={v['sign_test_p']}")
    for k, v in dd.get("cross_model", {}).items():
        L.append(f"cross_model {k:12s} n={v['n']} greedy[{seen(v['greedy'])}] "
                 f"jev[{seen(v['jev_static'])}]")
    ot = dd["oracle_threshold_loo"]
    L.append(f"oracle-threshold LOO (grid={ot['grid_size']}) {seen(ot)}")
    L.append(f"grid in-sample best {dd['grid_insample_best']['meta']} -> "
             f"{seen(dd['grid_insample_best'])}")
    fs = dd["fallback_sensitivity"]
    L.append(f"fallback base  static {seen(fs['base_static'])}")
    L.append(f"fallback occam static {seen(fs['occam_static'])}")
    L.append(f"fallback base  LOO-RSI {seen(fs['base_loo_rsi'])}")
    L.append(f"fallback occam LOO-RSI {seen(fs['occam_loo_rsi'])}")
    L.append("")
    dq = payload["data_quality"]
    L.append("")
    L.append(f"=== DATA QUALITY (clean decision groups n={dq['n_decision_clean']}"
             f" of {dq['n_decision_all']}; excluded={dq['excluded']}) ===")
    for k, v in dq["clean"].items():
        if isinstance(v, dict) and "n_groups" in v:
            L.append(f"{k:14s} {seen(v)}")
    L.append("")
    L.append("=== HISTORICAL REPRODUCTION (6 frozen groups) ===")
    hr = payload["historical_reproduction"]
    L.append(f"frozen artefact: {hr['frozen_artifact']}")
    for k, v in hr["replayed"].items():
        L.append(f"replayed {k:12s} {seen(v)}")
    L.append("")
    L.append(cross_task_mod._fmt(payload["cross_task"]))
    return "\n".join(L)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(RESULTS))
    args = ap.parse_args()
    payload = run_all(Path(args.out))
    print(_fmt(payload))
