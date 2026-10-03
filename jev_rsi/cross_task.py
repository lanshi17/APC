"""Phase 4 -- cross-task transfer meta-learning (RSI over transfer sources).

Implements the plan's ``scripts/cross_task_meta_learning.py`` /
``apc/decision/cross_task_rsi.py`` deliverable against the *frozen* APC
corpus.  No new runs and no API calls: every number is read from
``APC/experiments/apc_full_dataset.json``.

What a "transfer cell" is
-------------------------
A cell is one (target, source) pair with a measured outcome:

* **champion cells** -- the ``*-champ`` arms inside the multi-arm GPQA/HLE
  groups.  ``math-champ`` is the math-domain champion prompt deployed on the
  GPQA/HLE target; the reference arm is ``z0`` (no compiled artefact).
* **warm-start cells** -- the ``transfer-ws`` arm inside the ``transfer_*``
  groups; the reference arm is ``transfer-0`` (identity lineage).

Two tasks are evaluated
-----------------------
1. **Gain prediction** (per cell): predict ``hold(transfer) - hold(reference)``
   for a held-out cell.  Headline metric = MAE, the plan's ``MAE < 0.15``
   pre-registered criterion.
2. **Source selection** (per target): choose among
   ``{none} U available sources`` where ``none`` = do not transfer (gain 0).
   Headline metrics = exact best-source accuracy and deployment regret.

Protocols
---------
``leave_target`` -- train on cells whose target differs from the held-out
                   target (the honest cross-task protocol; no leakage of the
                   target's own difficulty).
``loo_cell``     -- per-cell leave-one-out (used for gain prediction).
``insample``     -- trains on every cell including the target (diagnostic
                   upper bound only; never quoted as a result).

Predictors: ``cold`` (predict gain 0 for every candidate), ``global_mean``,
``source_mean`` (**the RSI learner**: per-source mean of observed gains).
"""
from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from .data import Group, load_groups

RESULTS = Path(__file__).resolve().parent / "results"

CHAMPION_GROUPS = ("gpqa-s926", "hle-s921", "hle-s923")

# Source order fixes deterministic tie-breaking.
SOURCE_ORDER = ("math", "contract", "financial")
NONE = "none"
# Conservative tie-break: "none" first, then the declared source order.
CHOICE_ORDER = (NONE,) + SOURCE_ORDER

BASE_ARMS = ("z0", "transfer-0")


# ---------------------------------------------------------------------------
# Cell extraction
# ---------------------------------------------------------------------------
@dataclass
class TransferCell:
    cell_id: str
    kind: str            # "champion" | "warm-start"
    target: str          # group id
    target_task: str
    source: str
    transfer_arm: str
    base_arm: str
    base_val: float
    transfer_val: float
    base_hold: float
    transfer_hold: float
    val_gain: float
    gain: float
    val_usable: bool

    def as_dict(self) -> dict:
        d = asdict(self)
        for k in ("base_val", "transfer_val", "base_hold", "transfer_hold",
                  "val_gain", "gain"):
            d[k] = round(float(d[k]), 6)
        return d


def _source_of(arm_name: str) -> Optional[str]:
    for s in SOURCE_ORDER:
        if arm_name == f"{s}-champ":
            return s
    return None


def _parse_transfer_source(gid: str) -> str:
    body = gid[len("transfer_"):] if gid.startswith("transfer_") else gid
    body = body.split("-s")[0]
    return body.split("_to_")[0] if "_to_" in body else body


def _mk(kind: str, g: Group, source: str, transfer_arm: str,
        base, transfer) -> TransferCell:
    return TransferCell(
        cell_id=f"{g.gid}<={source}", kind=kind, target=g.gid,
        target_task=g.task, source=source, transfer_arm=transfer_arm,
        base_arm=base.name,
        base_val=float(base.val), transfer_val=float(transfer.val),
        base_hold=float(base.hold), transfer_hold=float(transfer.hold),
        val_gain=float(transfer.val - base.val),
        gain=float(transfer.hold - base.hold),
        val_usable=float(transfer.val) > 0.0,
    )


def extract_cells(groups: Sequence[Group]) -> List[TransferCell]:
    cells: List[TransferCell] = []
    for g in groups:
        if g.gid in CHAMPION_GROUPS:
            base = next((a for a in g.arms if a.name in BASE_ARMS), None)
            if base is None:
                continue
            for a in g.arms:
                src = _source_of(a.name)
                if src is not None:
                    cells.append(_mk("champion", g, src, a.name, base, a))
        elif g.family == "transfer":
            base = next((a for a in g.arms if a.name == "transfer-0"), None)
            ws = next((a for a in g.arms if a.name == "transfer-ws"), None)
            if base is not None and ws is not None:
                cells.append(_mk("warm-start", g, _parse_transfer_source(g.gid),
                                 "transfer-ws", base, ws))
    return sorted(cells, key=lambda c: (c.target, c.source))


# ---------------------------------------------------------------------------
# Prediction core
# ---------------------------------------------------------------------------
def predict(train: Sequence[TransferCell], candidates: Sequence[str],
            mode: str) -> Dict[str, float]:
    """Predicted gain for each candidate (``none`` is always exactly 0)."""
    if mode == "cold":
        return {s: 0.0 for s in candidates}
    gm = (sum(c.gain for c in train) / len(train)) if train else 0.0
    if mode == "global_mean":
        return {s: (0.0 if s == NONE else gm) for s in candidates}
    if mode == "source_mean":
        out: Dict[str, float] = {}
        for s in candidates:
            if s == NONE:
                out[s] = 0.0
                continue
            obs = [c.gain for c in train if c.source == s]
            out[s] = (sum(obs) / len(obs)) if obs else gm
        return out
    raise ValueError(mode)


def _choose(preds: Dict[str, float], candidates: Sequence[str]) -> str:
    return min(candidates,
               key=lambda s: (-preds[s], CHOICE_ORDER.index(s)
                              if s in CHOICE_ORDER else 99))


def _targets(cells: Sequence[TransferCell]) -> List[str]:
    return sorted({c.target for c in cells})


def _candidates(cells: Sequence[TransferCell], target: str) -> List[str]:
    srcs = sorted({c.source for c in cells if c.target == target},
                  key=lambda s: SOURCE_ORDER.index(s))
    return [NONE] + srcs


def _actual(cells: Sequence[TransferCell], target: str) -> Dict[str, float]:
    out = {NONE: 0.0}
    for c in cells:
        if c.target == target:
            out[c.source] = c.gain
    return out


def _train_pool(cells: Sequence[TransferCell], target: str,
                protocol: str, cell_id: Optional[str] = None):
    if protocol == "insample":
        return list(cells)
    if protocol == "leave_target":
        return [c for c in cells if c.target != target]
    if protocol == "loo_cell":
        return [c for c in cells if c.cell_id != cell_id]
    raise ValueError(protocol)


# ---------------------------------------------------------------------------
# Task 1 -- gain prediction (per cell)
# ---------------------------------------------------------------------------
def gain_prediction(cells: Sequence[TransferCell], protocol: str,
                    mode: str) -> dict:
    rows = []
    for c in cells:
        train = _train_pool(cells, c.target, protocol, c.cell_id)
        preds = predict(train, [c.source], mode)
        p = preds[c.source]
        rows.append({
            "cell": c.cell_id, "kind": c.kind, "target": c.target,
            "source": c.source,
            "predicted_gain": round(p, 6),
            "actual_gain": round(c.gain, 6),
            "abs_err": round(abs(p - c.gain), 6),
        })
    mae = sum(r["abs_err"] for r in rows) / len(rows) if rows else 0.0
    return {
        "n_cells": len(rows),
        "mae": round(mae, 6),
        "mae_champion_only": round(
            sum(r["abs_err"] for r in rows if r["kind"] == "champion")
            / max(1, sum(1 for r in rows if r["kind"] == "champion")), 6),
        "rows": rows,
    }


# ---------------------------------------------------------------------------
# Task 2 -- source selection (per target)
# ---------------------------------------------------------------------------
def source_selection(cells: Sequence[TransferCell], protocol: str,
                     mode: str) -> dict:
    rows = []
    for t in _targets(cells):
        cands = _candidates(cells, t)
        train = _train_pool(cells, t, protocol)
        preds = predict(train, cands, mode)
        chosen = _choose(preds, cands)
        act = _actual(cells, t)
        oracle = min(cands, key=lambda s: (-act[s], CHOICE_ORDER.index(s)))
        rows.append({
            "target": t,
            "kind": "champion" if t in CHAMPION_GROUPS else "warm-start",
            "candidates": cands,
            "predicted": {s: round(preds[s], 6) for s in cands},
            "chosen": chosen,
            "oracle": oracle,
            "correct": bool(chosen == oracle),
            "chosen_gain": round(act[chosen], 6),
            "oracle_gain": round(act[oracle], 6),
            "regret": round(act[oracle] - act[chosen], 6),
            "mae_chosen_calibration": round(abs(preds[chosen] - act[chosen]), 6),
        })
    n = len(rows)
    multi = [r for r in rows if len(r["candidates"]) > 2]
    return {
        "n_targets": n,
        "n_multi_candidate": len(multi),
        "accuracy": round(sum(r["correct"] for r in rows) / n, 4) if n else 0.0,
        "accuracy_multi": (round(sum(r["correct"] for r in multi) / len(multi), 4)
                           if multi else 0.0),
        "avg_regret": round(sum(r["regret"] for r in rows) / n, 6) if n else 0.0,
        "avg_regret_multi": (round(sum(r["regret"] for r in multi) / len(multi), 6)
                             if multi else 0.0),
        "mae_calibration": (round(sum(r["mae_chosen_calibration"] for r in rows) / n, 6)
                            if n else 0.0),
        "rows": rows,
    }


# ---------------------------------------------------------------------------
# Plan-facing API (CrossTaskRSI)
# ---------------------------------------------------------------------------
class CrossTaskRSI:
    """``should_use_transfer`` / ``update_after_transfer`` as in the plan."""

    def __init__(self, cells: Sequence[TransferCell]):
        self.cells = list(cells)
        self.observations: List[TransferCell] = []

    def update_after_transfer(self, cell: TransferCell) -> None:
        self.observations.append(cell)

    def should_use_transfer(self, target: str,
                            available_sources: Optional[Sequence[str]] = None
                            ) -> Tuple[Optional[str], float]:
        """Return ``(source_or_None, expected_gain)`` for one target group."""
        if available_sources is None:
            available_sources = sorted({c.source for c in self.cells
                                        if c.target == target},
                                       key=lambda s: SOURCE_ORDER.index(s))
        cands = [NONE] + [s for s in available_sources if s != NONE]
        pool = self.observations or [c for c in self.cells if c.target != target]
        preds = predict(pool, cands, "source_mean")
        best = _choose(preds, cands)
        if best == NONE:
            return None, 0.0
        return best, round(float(preds[best]), 6)


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------
PROTOCOLS = ("leave_target", "loo_cell", "insample")
MODES = ("cold", "global_mean", "source_mean")


def run(groups: Optional[Sequence[Group]] = None) -> dict:
    cells = extract_cells(groups if groups is not None else load_groups())

    protocols: Dict[str, dict] = {}
    for p in PROTOCOLS:
        protocols[p] = {
            "gain_prediction": {m: gain_prediction(cells, p, m) for m in MODES},
            "source_selection": {m: source_selection(cells, p, m) for m in MODES},
        }

    rsi = CrossTaskRSI(cells)
    decisions = []
    for t in _targets(cells):
        src, gain = rsi.should_use_transfer(t)
        decisions.append({
            "target": t,
            "candidates": [s for s in _candidates(cells, t) if s != NONE],
            "recommended_source": src,
            "predicted_gain": gain,
        })

    matrix = [
        {"target": t,
         "sources": {s: round(v, 6) for s, v in _actual(cells, t).items()}}
        for t in _targets(cells)
    ]

    lt = protocols["leave_target"]
    headline = {
        "protocol": "leave_target",
        "predictor": "source_mean(RSI)",
        "gain_mae": lt["gain_prediction"]["source_mean"]["mae"],
        "gain_mae_cold": lt["gain_prediction"]["cold"]["mae"],
        "gain_mae_target_met": lt["gain_prediction"]["source_mean"]["mae"] < 0.15,
        "selection_accuracy_multi": lt["source_selection"]["source_mean"]["accuracy_multi"],
        "selection_accuracy_multi_cold": lt["source_selection"]["cold"]["accuracy_multi"],
        "selection_regret_multi": lt["source_selection"]["source_mean"]["avg_regret_multi"],
        "selection_regret_multi_cold": lt["source_selection"]["cold"]["avg_regret_multi"],
    }

    return {
        "n_cells": len(cells),
        "n_targets": len(_targets(cells)),
        "val_usable": {
            "n_cells": len(cells),
            "n_val_usable": sum(1 for c in cells if c.val_usable),
            "n_champion_val_usable": sum(1 for c in cells
                                         if c.kind == "champion" and c.val_usable),
            "n_champion": sum(1 for c in cells if c.kind == "champion"),
            "n_warmstart_val_usable": sum(1 for c in cells
                                          if c.kind == "warm-start" and c.val_usable),
            "n_warmstart": sum(1 for c in cells if c.kind == "warm-start"),
        },
        "cells": [c.as_dict() for c in cells],
        "transfer_matrix": matrix,
        "protocols": protocols,
        "rsi_decisions": decisions,
        "headline": headline,
    }


def _fmt(payload: dict) -> str:
    L = [f"=== CROSS-TASK TRANSFER ({payload['n_cells']} cells / "
         f"{payload['n_targets']} targets) ==="]
    vu = payload["val_usable"]
    L.append(f"  val-score usable at decision time: {vu['n_val_usable']}/{vu['n_cells']} "
             f"(champion {vu['n_champion_val_usable']}/{vu['n_champion']}, "
             f"warm-start {vu['n_warmstart_val_usable']}/{vu['n_warmstart']})")
    L.append("--- measured matrix (holdout gain vs reference arm; none=0) ---")
    for m in payload["transfer_matrix"]:
        srcs = "  ".join(f"{k}:{v:+.4f}" for k, v in m["sources"].items())
        L.append(f"  {m['target']:38s} {srcs}")
    for p in ("leave_target", "insample"):
        gp = payload["protocols"][p]["gain_prediction"]
        ss = payload["protocols"][p]["source_selection"]
        L.append(f"--- protocol {p} ---")
        for mode in MODES:
            L.append(f"  gain-MAE  {mode:12s} {gp[mode]['mae']:.5f} "
                     f"(champ {gp[mode]['mae_champion_only']:.5f})")
        for mode in MODES:
            s = ss[mode]
            L.append(f"  select    {mode:12s} acc {s['accuracy_multi']:.2f} "
                     f"({s['n_multi_candidate']} multi) regret "
                     f"{s['avg_regret_multi']:+.5f} calib-MAE {s['mae_calibration']:.5f}")
    L.append("--- RSI source recommendations (leave-target-out) ---")
    for d in payload["rsi_decisions"]:
        rec = d["recommended_source"] or "none(no transfer)"
        L.append(f"  {d['target']:38s} -> {rec:10s} (pred {d['predicted_gain']:+.5f})")
    h = payload["headline"]
    L.append(f"HEADLINE: gain-MAE={h['gain_mae']:.5f} (cold {h['gain_mae_cold']:.5f}; "
             f"<0.15 = {h['gain_mae_target_met']}); "
             f"selection acc={h['selection_accuracy_multi']:.2f} "
             f"(cold {h['selection_accuracy_multi_cold']:.2f}); "
             f"regret={h['selection_regret_multi']:+.5f} "
             f"(cold {h['selection_regret_multi_cold']:+.5f})")
    return "\n".join(L)


if __name__ == "__main__":
    p = run()
    print(_fmt(p))
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "cross_task_results.json").write_text(
        json.dumps(p, indent=2, ensure_ascii=False))
    print(f"\nwrote {RESULTS / 'cross_task_results.json'}")
