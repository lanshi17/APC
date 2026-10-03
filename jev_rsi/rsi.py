"""Recursive Self-Improvement (RSI) over the Jev gate's meta-parameters.

Protocol per RSI round:

1. **Trace** -- every deployment decision is logged together with the features
   the gate saw, the rule that fired, and the *post-hoc* regret.
2. **Failure clustering** -- traces whose regret exceeds the noise band are
   embedded in a small standardised feature space and clustered (deterministic
   k-means).  Each cluster contributes *one* vote, so a single repeated failure
   mode cannot dominate the update the way raw per-trace voting would.
3. **Meta-parameter update** -- each failure signature votes for a bounded
   adjustment of one threshold; the step size decays geometrically.

The learner never sees holdout scores of the groups it will later be evaluated
on: training and evaluation groups are disjoint in every reported protocol.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .core import NOISE_BAND, Decision, JevGate, MetaParams
from .data import Group


# ---------------------------------------------------------------------------
# Trace records
# ---------------------------------------------------------------------------
@dataclass
class Trace:
    gid: str
    family: str
    reason: str
    deployed: str
    regret: float
    signature: str
    features: Dict[str, float]

    def as_dict(self) -> dict:
        d = asdict(self)
        d["regret"] = round(self.regret, 6)
        return d


def failure_signature(g: Group, d: Decision) -> str:
    """Classify a decision outcome into one of the RSI failure modes."""
    if d.regret < NOISE_BAND:
        return "success"
    if g.has_collapse and d.deployed_key != g.base.key:
        return "collapse_miss"
    if d.reason == "normal_selection":
        return "val_overestimate"
    if d.reason in ("low_confidence_occam", "low_val_alignment",
                    "collapse_detected", "always_conservative"):
        return "over_conservative"
    return "other"


# ---------------------------------------------------------------------------
# Deterministic k-means (no sklearn in this environment)
# ---------------------------------------------------------------------------
FEATURE_KEYS = ("margin", "collapse_noul", "alignment", "confidence", "n_arms")


def _kmeans(X: np.ndarray, k: int, seed: int = 7, iters: int = 50) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n = len(X)
    k = max(1, min(k, n))
    # k-means++ init, seeded
    centers = [X[rng.integers(n)]]
    for _ in range(1, k):
        d2 = np.min([np.sum((X - c) ** 2, axis=1) for c in centers], axis=0)
        probs = d2 / d2.sum() if d2.sum() > 0 else np.full(n, 1 / n)
        centers.append(X[rng.choice(n, p=probs)])
    C = np.array(centers)
    labels = np.zeros(n, dtype=int)
    for _ in range(iters):
        dist = np.linalg.norm(X[:, None, :] - C[None, :, :], axis=2)
        new = np.argmin(dist, axis=1)
        if np.array_equal(new, labels):
            break
        labels = new
        for j in range(k):
            m = labels == j
            if m.any():
                C[j] = X[m].mean(axis=0)
    return labels


# ---------------------------------------------------------------------------
# The learner
# ---------------------------------------------------------------------------
VOTE = {
    "collapse_miss": ("collapse_threshold", -1),
    "val_overestimate": ("alignment_threshold", +1),
    "over_conservative_low_confidence_occam": ("confidence_floor", -1),
    "over_conservative_low_val_alignment": ("alignment_threshold", -1),
    "over_conservative_collapse_detected": ("collapse_threshold", +1),
    "over_conservative_always_conservative": ("confidence_floor", -1),
}


class RSILearner:
    def __init__(self, meta: Optional[MetaParams] = None, clustering: bool = True,
                 step: float = 0.05, min_step: float = 0.01, buffer: int = 3,
                 vote_cap: int = 2):
        self.meta = meta or MetaParams()
        self.clustering = clustering
        self.step0 = step
        self.min_step = min_step
        self.buffer = buffer
        self.vote_cap = vote_cap          # max threshold moves per round
        self.traces: List[Trace] = []
        self.history: List[dict] = []
        self.consumed: set = set()
        self.rounds = 0

    # -- observation ------------------------------------------------------
    def observe(self, g: Group, d: Decision) -> Optional[dict]:
        sig = failure_signature(g, d)
        self.traces.append(Trace(gid=g.gid, family=g.family, reason=d.reason,
                                 deployed=d.deployed_name, regret=d.regret,
                                 signature=sig, features=dict(d.features)))
        pending = [t for t in self.traces if t.gid not in self.consumed]
        if len(pending) >= self.buffer:
            return self.update()
        return None

    # -- update -----------------------------------------------------------
    def update(self) -> Optional[dict]:
        pending = [t for t in self.traces if t.gid not in self.consumed]
        failures = [t for t in pending if t.signature != "success"]
        if not failures:
            # consume the buffer, nothing to learn from clean decisions
            for t in pending:
                self.consumed.add(t.gid)
                self.history.append({"gid": t.gid, "signature": t.signature})
            return None

        votes = self._votes(failures)
        step = max(self.min_step, self.step0 / (1 + 0.5 * self.rounds))
        before = self.meta.as_dict()
        moves: Dict[str, float] = {}
        for param, direction in votes:
            moves[param] = moves.get(param, 0.0) + direction * step
        for param, delta in moves.items():
            setattr(self.meta, param, getattr(self.meta, param) + delta)
        self.meta = self.meta.clamped()
        self.rounds += 1

        for t in self.traces:
            if t.gid not in self.consumed:
                self.consumed.add(t.gid)
                self.history.append({"gid": t.gid, "signature": t.signature,
                                     "regret": round(t.regret, 6)})
        rec = {
            "round": self.rounds,
            "n_failures": len(failures),
            "signatures": _count(failures),
            "before": before,
            "after": self.meta.as_dict(),
            "moves": {k: round(v, 5) for k, v in moves.items()},
            "step": round(step, 5),
        }
        self.history.append(rec)
        return rec

    def _votes(self, failures: List[Trace]) -> List[Tuple[str, int]]:
        if not self.clustering:
            raw = [self._vote_of(t) for t in failures]
            raw = [v for v in raw if v]
            return raw[: self.vote_cap]

        X = np.array([[t.features.get(k, 0.0) for k in FEATURE_KEYS]
                      for t in failures], dtype=float)
        X = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-9)
        k = 2 if len(failures) >= 4 else 1
        labels = _kmeans(X, k)
        votes: List[Tuple[str, int]] = []
        for j in range(int(labels.max()) + 1):
            members = [t for t, l in zip(failures, labels) if l == j]
            if not members:
                continue
            majority = _majority_signature(members)
            t = next(t for t in members if t.signature == majority)
            v = self._vote_of(t)
            if v:
                votes.append(v)
        return votes[: self.vote_cap]

    @staticmethod
    def _vote_of(t: Trace) -> Optional[Tuple[str, int]]:
        if t.signature == "over_conservative":
            key = f"over_conservative_{t.reason}"
            return VOTE.get(key)
        return VOTE.get(t.signature)


def _majority_signature(ts: Sequence[Trace]) -> str:
    c = _count(ts)
    return max(c.items(), key=lambda kv: (kv[1], kv[0]))[0]


def _count(ts: Sequence[Trace]) -> Dict[str, int]:
    out: Dict[str, int] = {}
    for t in ts:
        out[t.signature] = out.get(t.signature, 0) + 1
    return out


# ---------------------------------------------------------------------------
# Replay driver
# ---------------------------------------------------------------------------
def replay(train_groups: Sequence[Group], clustering: bool = True,
           meta0: Optional[MetaParams] = None,
           enabled: Sequence[str] = JevGate.ALL,
           fallback: str = "base") -> RSILearner:
    """Run the RSI loop over an ordered list of *training* groups."""
    learner = RSILearner(meta=meta0, clustering=clustering)
    gate = JevGate(meta=learner.meta, enabled=enabled, fallback=fallback)
    for g in train_groups:
        gate.meta = learner.meta
        d = gate.act(g)
        learner.observe(g, d)
    # flush any residual buffer
    learner.buffer = 1
    learner.update()
    return learner
