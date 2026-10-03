"""Dataset loading + normalisation for the Jev + RSI experiment harness.

Single source of truth is the frozen APC artefact
``experiments/apc_full_dataset.json`` inside the (read-only) APC code repo.
Nothing here fabricates numbers: every arm score is read from that file, which
itself was assembled from real benchmark run files under
``experiments/apcbench/``.

Arm scores are *validation* (8-point val set) and *holdout* scores.  The gate
never sees holdout at decision time; holdout is only used to score the decision
after the fact.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

# Location-robust root resolution: the harness works both inside the paper
# workspace (``apc_paper/jev_rsi`` with the dataset at ``apc_paper/APC/...``)
# and directly inside the APC code repo (``02_APC/jev_rsi`` with the dataset at
# ``02_APC/experiments/...``).  A stale ``APC`` sub-directory must not shadow
# the real dataset.
_HOME = Path(__file__).resolve().parents[1]
PAPER_ROOT = _HOME
APC_ROOT = (_HOME / "APC"
            if (_HOME / "APC" / "experiments" / "apc_full_dataset.json").exists()
            else _HOME)
DATASET = APC_ROOT / "experiments" / "apc_full_dataset.json"

# ---------------------------------------------------------------------------
# "Occam order": which arm of a family counts as the conservative / simplest
# deployment.  This mirrors the declared order used by the original
# AutoAPC-Select gate (scripts/auto_apc_gate.py) so that the heuristic baseline
# is reproduced faithfully rather than re-invented.
# ---------------------------------------------------------------------------
FAMILY_ORDER: Dict[str, List[str]] = {
    "apc": ["zero-shot", "manual", "apc-safe", "apc-full"],
    "transfer": ["transfer-ws", "transfer-0", "cold"],
    "gpqa": ["z0", "math-champ", "contract-champ", "financial-champ", "espo", "gepa"],
    "hle": ["z0", "math-champ", "contract-champ", "financial-champ", "gepa", "espo"],
    "hle_ablation": [
        "base", "minus-role", "minus-output", "minus-verification",
        "minus-reasoning", "minus-layout", "minus-constraints",
    ],
    "gepa": ["gepa", "espo"],
    "default": ["z0", "zero-shot", "z0-control", "reeval-z0", "z0-paired",
                "manual", "reeval-manual", "apc-safe", "reeval-safe", "safe",
                "transfer-0", "cold", "apc-full", "reeval-full",
                "gepa", "reeval-gepa", "espo"],
}


# Arms that apply no compiled artefact / no transfer -- the gate's safe harbour.
BASE_ARM_NAMES = {
    "zero-shot", "z0", "z0-control", "z0-paired", "reeval-z0",
    "cold", "transfer-0", "t0", "base", "pgam-cold",
}


def family_of(group_id: str, task: str) -> str:
    """Map a group id onto the Occam-order family whose declared order applies."""
    gid = group_id.lower()
    if "transfer" in gid:
        return "transfer"
    if gid.startswith("gpqa"):
        return "gpqa"
    if gid.startswith("hle_ablation"):
        return "hle_ablation"
    if gid.startswith("hle"):
        return "hle"
    if gid.startswith("gepa"):
        return "gepa"
    if task in ("financial", "contract", "math"):
        return "apc"
    return "default"


@dataclass
class Arm:
    key: str                 # unique within a group (method names repeat)
    name: str
    val: float
    hold: float
    rank: int                # lower = more conservative per Occam order
    simplicity: int

    def as_dict(self) -> dict:
        return {"key": self.key, "name": self.name, "val": self.val,
                "hold": self.hold, "rank": self.rank}


@dataclass
class Group:
    gid: str
    task: str
    family: str
    arms: List[Arm] = field(default_factory=list)

    # ---- derived ---------------------------------------------------------
    @property
    def n_arms(self) -> int:
        return len(self.arms)

    @property
    def oracle(self) -> Arm:
        return max(self.arms, key=lambda a: a.hold)

    @property
    def greedy(self) -> Arm:
        """argmax val, ties broken by Occam rank then row order."""
        return min(self.arms, key=lambda a: (-a.val, a.rank))

    @property
    def conservative(self) -> Arm:
        """The family's declared conservative arm (Occam minimum, best val)."""
        return min(self.arms, key=lambda a: (a.rank, -a.val))

    @property
    def base(self) -> Arm:
        """The *no-adaptation* arm: the untouched / zero-transfer deployment.

        The original JevGate hard-codes ``'z0'`` as the fallback for the
        collapse and alignment rules.  That literal name does not exist in the
        transfer / gepa / gpqa / hle groups, so the faithful generalisation is
        "the arm that applies no compiled artefact", i.e. zero-shot / cold /
        zero-adaptation.  When a group has no such arm we fall back to the
        family's Occam-minimum arm.
        """
        cand = [a for a in self.arms if a.name in BASE_ARM_NAMES]
        if cand:
            return max(cand, key=lambda a: (a.val, -a.rank))
        return self.conservative

    @property
    def is_decision(self) -> bool:
        """A real choice exists only when >1 arm survived validation."""
        return len(self.arms) >= 2

    @property
    def is_informative(self) -> bool:
        """The naive val-argmax rule leaves measurable regret on the table."""
        return self.is_decision and (self.oracle.hold - self.greedy.hold) > 1e-9

    @property
    def has_collapse(self) -> bool:
        """A genuinely collapsed arm: catastrophically low val *and* holdout."""
        if len(self.arms) < 2:
            return False
        best = max(a.hold for a in self.arms)
        return any(a.val < 0.35 and (best - a.hold) > 0.20 for a in self.arms)


def _rank(name: str, family: str) -> int:
    order = FAMILY_ORDER.get(family, FAMILY_ORDER["default"])
    if name in order:
        return order.index(name)
    # Unknown arm -> fall back to the global default order, else most complex.
    dflt = FAMILY_ORDER["default"]
    return 100 + (dflt.index(name) if name in dflt else len(dflt))


def load_groups(dataset: Path = DATASET) -> List[Group]:
    raw = json.loads(dataset.read_text())
    groups: List[Group] = []
    for gid, rec in raw["tasks"].items():
        arms: List[Arm] = []
        for i, a in enumerate(rec["arms"]):
            val = a.get("val_score")
            hold = a.get("holdout_score")
            if val is None or hold is None:
                continue                      # unscorable row (e.g. gepa_nt)
            fam = family_of(gid, rec.get("task", ""))
            arms.append(Arm(key=f"{a['name']}#{i}", name=a["name"],
                            val=float(val), hold=float(hold),
                            rank=_rank(a["name"], fam),
                            simplicity=_rank(a["name"], "default")))
        if arms:
            fam = family_of(gid, rec.get("task", ""))
            groups.append(Group(gid=gid, task=rec.get("task", ""),
                                family=fam, arms=arms))
    return sorted(groups, key=lambda g: g.gid)


def split_key(gid: str) -> str:
    """Stratification key used by the leave-one-out / holdout protocols."""
    return gid.split("-s")[0]
