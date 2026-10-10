"""P5 -- stronger meta-learner: Bayesian optimisation over the gate thresholds.

``FUTURE_DIRECTIONS.md`` Priority 5 argues that the 175-point grid search is
too weak and proposes a Gaussian-process surrogate over
``(collapse_threshold, alignment_threshold, confidence_floor)`` with the mean
training regret as the objective, evaluated leave-one-group-out.

This module runs that proposal, and to keep the comparison honest it also runs

* the same search box as the existing grid, not just the document's narrower box;
* a **random search at an identical evaluation budget**, so that "BO beats the
  grid" is not just "50 evaluations beat 175" -- or "more evaluations help";
* the in-sample optimum of each method, for the optimism gap.

The GP is a plain RBF surrogate written against numpy/scipy (no scikit-learn),
with the length scale and noise chosen by a coarse grid over the log marginal
likelihood.  Everything is seeded and deterministic.

Usage
-----
    python -m jev_rsi.bayes_meta
    python -m jev_rsi.bayes_meta --out /tmp/x
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from numpy.linalg import slogdet
from scipy.linalg import cho_factor, cho_solve
from scipy.stats import norm, qmc

from .core import Decision, JevGate, MetaParams
from .data import Group
from .experiments import Harness, summarize

DEFAULT_OUT = Path(__file__).resolve().parent / "results"
KEYS = ("collapse_threshold", "alignment_threshold", "confidence_floor")

# the grid search's own box, and the narrower one the document proposes
BOXES: Dict[str, List[Tuple[float, float]]] = {
    "grid_box": [(0.20, 0.90), (0.05, 0.70), (0.05, 0.80)],
    "document_box": [(0.50, 0.90), (0.30, 0.70), (0.50, 0.90)],
}
N_INIT = 12
BUDGET = 50          # total evaluations, matched across BO and random search
N_CANDIDATES = 2048  # acquisition-pool size


# ---------------------------------------------------------------------------
# objective
# ---------------------------------------------------------------------------
def _decisions(h: Harness, groups: Sequence[Group], meta: MetaParams) -> List[Decision]:
    return h.evaluate("jev", groups, meta=meta)


def mean_regret(h: Harness, groups: Sequence[Group], meta: MetaParams) -> float:
    ds = _decisions(h, groups, meta)
    return float(np.mean([d.regret for d in ds]))


# ---------------------------------------------------------------------------
# minimal GP with EI acquisition
# ---------------------------------------------------------------------------
def _sqdist(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    return (np.sum(A ** 2, axis=1)[:, None] + np.sum(B ** 2, axis=1)[None, :]
            - 2.0 * A @ B.T)


def _gp_posterior(X: np.ndarray, y: np.ndarray, Xq: np.ndarray,
                  length_scales: Sequence[float],
                  noises: Sequence[float]) -> Tuple[np.ndarray, np.ndarray, float, float]:
    """Return (mean, var, best_length_scale, best_noise) at ``Xq``.

    Inputs are normalised to [0, 1] and the targets standardised, so a single
    isotropic length scale per candidate is meaningful.  Hyper-parameters come
    from a coarse grid over the log marginal likelihood.
    """
    y_mean, y_std = float(y.mean()), float(y.std() + 1e-12)
    ys = (y - y_mean) / y_std
    sf2 = float(np.var(ys) + 1e-9)
    D = _sqdist(X, X)

    best_lml, best = -np.inf, None
    for l in length_scales:
        K = sf2 * np.exp(-0.5 * D / (l ** 2))
        for sn2 in noises:
            Ky = K + sn2 * np.eye(len(X))
            try:
                c = cho_factor(Ky, lower=True)
            except np.linalg.LinAlgError:
                continue
            alpha = cho_solve(c, ys)
            sign, logdet = slogdet(Ky)
            if sign <= 0:
                continue
            lml = -0.5 * float(ys @ alpha) - 0.5 * logdet - 0.5 * len(X) * np.log(2 * np.pi)
            if lml > best_lml:
                best_lml, best = lml, (l, sn2, K, c)
    if best is None:
        return (np.full(len(Xq), y_mean), np.full(len(Xq), y_std ** 2), 0.0, 0.0)

    l, sn2, K, c = best
    Ks = sf2 * np.exp(-0.5 * _sqdist(X, Xq) / (l ** 2))
    mu = Ks.T @ cho_solve(c, ys)
    v = cho_solve(c, Ks)
    var = np.maximum(sf2 - np.sum(Ks * v, axis=0), 1e-12)
    return mu * y_std + y_mean, var * y_std ** 2, float(l), float(sn2)


def _expected_improvement(mu: np.ndarray, var: np.ndarray, best: float) -> np.ndarray:
    sigma = np.sqrt(np.maximum(var, 1e-12))
    z = (best - mu) / sigma
    return (best - mu) * norm.cdf(z) + sigma * norm.pdf(z)


# ---------------------------------------------------------------------------
# search strategies, all on the unit cube
# ---------------------------------------------------------------------------
def _to_meta(u: np.ndarray, box: Sequence[Tuple[float, float]]) -> MetaParams:
    lo = np.array([b[0] for b in box]); hi = np.array([b[1] for b in box])
    vals = lo + np.clip(u, 0.0, 1.0) * (hi - lo)
    return MetaParams(*[float(v) for v in vals]).clamped()


def _sobol_points(seed: int, n: int) -> np.ndarray:
    """Sobol points, drawn in a power-of-two block and sliced to ``n``.

    The balance guarantee only holds for power-of-two counts; slicing a larger
    block keeps the low-discrepancy property and avoids the numpy warning.
    """
    block = 1 << max(1, int(np.ceil(np.log2(max(n, 2)))))
    return qmc.Sobol(d=3, scramble=True, seed=seed).random(block)[:n]


def _unit(u: np.ndarray, box: Sequence[Tuple[float, float]]) -> np.ndarray:
    lo = np.array([b[0] for b in box]); hi = np.array([b[1] for b in box])
    return (u - lo) / (hi - lo)


def bayes_search(h: Harness, train: Sequence[Group], box, budget: int = BUDGET,
                 n_init: int = N_INIT, seed: int = 20261003) -> Dict[str, Any]:
    """GP + EI minimisation of the mean training regret."""
    draw = _sobol_points(seed, BUDGET + N_CANDIDATES)
    X = [np.array(u) for u in draw[:n_init]]        # points already live on [0,1]^3
    y = [mean_regret(h, train, _to_meta(x, box)) for x in X]
    pool = draw[BUDGET:BUDGET + N_CANDIDATES]       # disjoint from the evaluated points

    trace = [{"phase": "init", "u": X[i].tolist(), "obj": y[i]} for i in range(n_init)]
    while len(X) < budget:
        Xa, ya = np.array(X), np.array(y)
        mu, var, l, sn2 = _gp_posterior(
            Xa, ya, pool,
            length_scales=[0.05, 0.1, 0.2, 0.35, 0.5, 0.75, 1.0, 1.5],
            noises=[1e-6, 1e-4, 1e-3, 1e-2, 1e-1],
        )
        u_next = pool[int(np.argmax(_expected_improvement(mu, var, float(ya.min()))))]
        X.append(np.array(u_next))
        y.append(mean_regret(h, train, _to_meta(u_next, box)))
        trace.append({"phase": "bo", "u": u_next.tolist(), "obj": y[-1],
                      "length_scale": l, "noise": sn2})

    i_best = int(np.argmin(y))
    return {"best_u": X[i_best].tolist(), "best_meta": _to_meta(X[i_best], box).as_dict(),
            "best_obj": y[i_best], "n_evals": len(X), "trace": trace}


def random_search(h: Harness, train: Sequence[Group], box, budget: int = BUDGET,
                  seed: int = 20261004) -> Dict[str, Any]:
    us = _sobol_points(seed, budget)
    ys = [mean_regret(h, train, _to_meta(u, box)) for u in us]
    i = int(np.argmin(ys))
    return {"best_u": np.array(us[i]).tolist(), "best_meta": _to_meta(us[i], box).as_dict(),
            "best_obj": ys[i], "n_evals": len(us), "all_objs": [float(v) for v in ys]}


def _meta_to_vec(meta: MetaParams) -> List[float]:
    return [getattr(meta, k) for k in KEYS]


# ---------------------------------------------------------------------------
# leave-one-group-out harness
# ---------------------------------------------------------------------------
def loo_strategy(h: Harness, strategy: str, box_name: str,
                 budget: int = BUDGET) -> Dict[str, Any]:
    groups = h.decision_groups
    static = MetaParams()
    rows = []
    for i, test in enumerate(groups):
        train = [g for j, g in enumerate(groups) if j != i]
        if strategy == "bayes":
            best = bayes_search(h, train, BOXES[box_name], budget=budget)
        elif strategy == "random":
            best = random_search(h, train, BOXES[box_name], budget=budget)
        else:
            raise ValueError(strategy)
        meta = MetaParams(**best["best_meta"])
        d = JevGate(meta=meta).act(test, h.feats[test.gid])
        s = JevGate(meta=static).act(test, h.feats[test.gid])
        rows.append({"gid": test.gid, "meta": best["best_meta"],
                     "deployed": d.deployed_name, "reason": d.reason,
                     "regret": round(d.regret, 6),
                     "static_regret": round(s.regret, 6)})
    ds = [Decision(r["gid"], r["gid"], "learned", hold=0.0, regret=r["regret"]) for r in rows]
    return {
        "strategy": strategy, "box": box_name, "budget": budget,
        "summary": summarize(groups, ds),
        "vs_static": {
            "wins": sum(1 for r in rows if r["static_regret"] - r["regret"] > 1e-9),
            "losses": sum(1 for r in rows if r["regret"] - r["static_regret"] > 1e-9),
            "ties": sum(1 for r in rows if abs(r["regret"] - r["static_regret"]) <= 1e-9),
        },
        "rows": rows,
    }


def in_sample_best(h: Harness, strategy: str, box_name: str,
                   budget: int = BUDGET) -> Dict[str, Any]:
    groups = h.decision_groups
    if strategy == "bayes":
        best = bayes_search(h, groups, BOXES[box_name], budget=budget)
    elif strategy == "random":
        best = random_search(h, groups, BOXES[box_name], budget=budget)
    else:
        raise ValueError(strategy)
    meta = MetaParams(**best["best_meta"])
    return {"strategy": strategy, "box": box_name,
            "meta": best["best_meta"], "best_obj": best["best_obj"],
            **summarize(groups, h.evaluate("jev", groups, meta=meta))}


def landscape_granularity(h: Harness, box_name: str = "grid_box",
                          n: int = 400, seed: int = 20261005) -> Dict[str, Any]:
    """How coarse is the objective?  A step function defeats a smooth surrogate."""
    groups = h.decision_groups
    objs, decision_vectors = [], set()
    for u in _sobol_points(seed, n):
        meta = _to_meta(u, BOXES[box_name])
        ds = h.evaluate("jev", groups, meta=meta)
        objs.append(round(float(np.mean([d.regret for d in ds])), 6))
        decision_vectors.add(tuple(d.deployed_key for d in ds))
    return {
        "n_samples": n, "box": box_name,
        "n_distinct_objectives": len(set(objs)),
        "n_distinct_decision_vectors": len(decision_vectors),
        "objective_min": min(objs), "objective_max": max(objs),
        "objective_median": float(np.median(objs)),
    }


def build_report() -> Dict[str, Any]:
    h = Harness(corpus="frozen")
    groups = h.decision_groups
    greedy = summarize(groups, h.evaluate("greedy", groups))

    report: Dict[str, Any] = {
        "protocol": ("FUTURE_DIRECTIONS.md Priority 5: GP surrogate over the three gate "
                     "thresholds, mean training regret, leave-one-group-out"),
        "budget": BUDGET, "n_init": N_INIT,
        "boxes": {k: [list(b) for b in v] for k, v in BOXES.items()},
        "greedy": greedy,
        "existing_grid_loo": _existing_grid(h),
        "landscape": landscape_granularity(h),
        "strategies": {},
    }
    for box_name in BOXES:
        report["strategies"][box_name] = {
            "bayes_loo": loo_strategy(h, "bayes", box_name),
            "random_loo": loo_strategy(h, "random", box_name),
            "bayes_in_sample": in_sample_best(h, "bayes", box_name),
            "random_in_sample": in_sample_best(h, "random", box_name),
        }
    return report


def _existing_grid(h: Harness) -> Dict[str, Any]:
    """The 175-point grid as already published, so the comparison is like for like."""
    from .experiments import GRID
    groups = h.decision_groups
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
        rows.append({"gid": test.gid, "regret": round(d.regret, 6)})
    ds = [Decision(r["gid"], r["gid"], "grid", hold=0.0, regret=r["regret"]) for r in rows]
    return {**summarize(groups, ds), "n_points": 7 * 5 * 5}


def _print(rep: Dict[str, Any]) -> None:
    g = rep["greedy"]
    print(f"baseline greedy: {g['exact_oracle']}/32 exact, within-noise {g['within_noise']}/32, "
          f"avg regret {g['avg_regret']:+.5f}")
    eg = rep["existing_grid_loo"]
    print(f"existing 175-point grid (LOO): {eg['exact_oracle']}/32 exact, "
          f"avg regret {eg['avg_regret']:+.5f}")
    L = rep["landscape"]
    print(f"\nlandscape over {L['n_samples']} samples in {L['box']}: "
          f"{L['n_distinct_objectives']} distinct objective values, "
          f"{L['n_distinct_decision_vectors']} distinct decision vectors")
    for box_name, s in rep["strategies"].items():
        print(f"\n=== box: {box_name} {rep['boxes'][box_name]} ===")
        print(f"  {'method':<22} {'LOO exact':>9} {'within':>7} {'avg regret':>11}  {'W/L/T':>9}  {'in-sample':>9}")
        for name, loo_key, ins_key in (("Bayesian opt. (50)", "bayes_loo", "bayes_in_sample"),
                                       ("random search (50)", "random_loo", "random_in_sample")):
            l = s[loo_key]; ins = s[ins_key]
            wlt = f"{l['vs_static']['wins']}/{l['vs_static']['losses']}/{l['vs_static']['ties']}"
            print(f"  {name:<22} {l['summary']['exact_oracle']:>6}/32 {l['summary']['within_noise']:>5}/32 "
                  f"{l['summary']['avg_regret']:>+11.5f}  {wlt:>9}  {ins['exact_oracle']:>6}/32")


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    args = ap.parse_args(argv)
    rep = build_report()
    _print(rep)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    path = out / "bayes_meta_results.json"
    path.write_text(json.dumps(rep, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nwrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
