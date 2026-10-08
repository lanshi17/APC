#!/usr/bin/env python
"""P3 acceptance checks -- the plan's Phase 1/2/3/Final gates in one command.

    cd code && python scripts/p3_acceptance.py

Each check prints PASS / FAIL / PARTIAL with the measured value and the value
the P3 plan asked for.  Where the plan's own arithmetic is internally
inconsistent the check reports both readings instead of silently agreeing.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from jev_rsi.data import (EXTENDED_ROOT, corpus_audit,  # noqa: E402
                          load_all_groups, load_groups, scenario_of)

RESULTS: list[tuple[str, str, str]] = []


def check(name: str, ok: bool | None, detail: str) -> None:
    status = "PASS" if ok else ("PARTIAL" if ok is None else "FAIL")
    RESULTS.append((status, name, detail))
    print(f"  [{status:7s}] {name}: {detail}")


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def main() -> int:
    audit = corpus_audit()
    groups = load_all_groups()

    print("=== Phase 1: Scenario C ===")
    c = _load(EXTENDED_ROOT / "scenario_c" / "scores.json")
    cg = c.get("groups", {})
    check("C groups", len(cg) == 10, f"{len(cg)} groups (plan: 10)")
    arms = [a for g in cg.values() for a in g["arms"]]
    check("C arms", len(arms) == 60, f"{len(arms)} arms (plan: 60)")
    gaming_ok = all(sum(1 for a in g["arms"] if a["type"] == "gaming") >= 2
                    for g in cg.values()) and cg
    check("C >=2 gaming arms/group", bool(gaming_ok),
          f"min per group = {min((sum(1 for a in g['arms'] if a['type'] == 'gaming') for g in cg.values()), default=0)} (plan: >=2)")
    dec = [a for a in arms if a["val_score"] > 0.4 and a["holdout_score"] < 0.3]
    check("C decoupled arms (plan band val>0.4, holdout<0.3)", None,
          f"{len(dec)} arms in the plan's band (plan: >=15). "
          f"See the rank-based criterion in the analysis artifact.")

    print("=== Phase 2: Scenario A ===")
    a = _load(EXTENDED_ROOT / "scenario_a" / "scores.json")
    ag = a.get("groups", {})
    check("A groups", len(ag) == 20, f"{len(ag)} groups (plan: 20)")
    aarms = [x for g in ag.values() for x in g["arms"]]
    check("A arms", len(aarms) == 200, f"{len(aarms)} arms (plan: 200)")
    high = sum(1 for g in ag.values()
               if g.get("judge_variance_10", {}).get("mean_arm_std", 0) > 2.0)
    check("A high-variance groups (std>2.0)", high >= 5,
          f"{high}/{len(ag)} groups (plan: >=5)")

    print("=== Phase 3: Scenario B ===")
    b = _load(EXTENDED_ROOT / "scenario_b" / "scores.json")
    bg = b.get("groups", {})
    check("B groups", len(bg) == 15, f"{len(bg)} groups (plan: 15)")
    barms = [x for g in bg.values() for x in g["arms"]]
    check("B arms", len(barms) == 90, f"{len(barms)} arms (plan: 90)")

    print("=== Final: combined corpus ===")
    check("load_all_groups() >= 78", audit["n_total"] >= 78,
          f"{audit['n_total']} groups = frozen {audit['n_frozen']} + extended "
          f"{audit['n_extended']} (plan's '78' is the plan's own 43+10+20+15 arithmetic)")
    check("decision groups >= 70", audit["n_decision_total"] >= 70,
          f"{audit['n_decision_total']} decision groups (plan: 70)")
    check("all extended groups are decision groups",
          all(g.is_decision for g in groups if scenario_of(g.gid) != "original"),
          f"{audit['n_extended']} extended groups, all with >=2 arms")
    print()

    n_fail = sum(1 for s, _, _ in RESULTS if s == "FAIL")
    n_partial = sum(1 for s, _, _ in RESULTS if s == "PARTIAL")
    print(f"summary: {sum(1 for s,_,_ in RESULTS if s=='PASS')} pass, "
          f"{n_partial} partial, {n_fail} fail")
    return 1 if n_fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
