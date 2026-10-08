"""Decision primitives and deployment policies.

Everything here is a *deterministic offline replay*: the gate sees only
``val_score`` and the benchmark's documented noise band, never ``holdout_score``.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .data import Arm, Group

NOISE_BAND = 0.007          # documented same-day validation noise band
ALIGN_SCALE = 3 * NOISE_BAND  # margin at which the val ranking is fully trusted
VAL_SIGMA = NOISE_BAND      # val-score sampling sigma used by the Bayesian Choice
N_MC = 20000                # Monte-Carlo draws per Choice question


# ---------------------------------------------------------------------------
# Primitives: Choice / Noul / Score  (deterministic, seeded)
# ---------------------------------------------------------------------------
def choice_posterior(arms: Sequence[Arm], sigma: float = VAL_SIGMA,
                     n_mc: int = N_MC, seed: int = 20261002) -> Dict[str, float]:
    """P(arm is truly best) under a Gaussian model of val-score noise.

    ``sigma`` is the observed validation noise band, so 0.007 of val margin
    corresponds to ~one standard error -> near-uniform posterior, which is the
    calibrated behaviour we want from the gate's confidence signal.
    """
    vals = np.array([a.val for a in arms], dtype=float)
    rng = np.random.default_rng(seed)
    draws = rng.normal(loc=vals[None, :], scale=sigma, size=(n_mc, len(arms)))
    winners = np.argmax(draws, axis=1)
    counts = np.bincount(winners, minlength=len(arms)).astype(float)
    probs = counts / counts.sum()
    return {a.key: float(p) for a, p in zip(arms, probs)}

def collapse_noul(arms: Sequence[Arm]) -> float:
    """Noul: evidence that a validated arm has suffered seed collapse (0..1).

    Continuous version of the piecewise rule in
    ``apc-pipeline/apc/decision/jev_client.py``.  Requires *both* a large
    spread across arms and a low absolute floor, so a uniformly mediocre group
    is not mistaken for a collapse.
    """
    if len(arms) < 2:
        return 0.0
    vals = [a.val for a in arms]
    spread = max(vals) - min(vals)
    low = _clip((0.5 - min(vals)) / 0.5)
    return float(low * _clip(spread / 0.5))


def alignment_score(arms: Sequence[Arm], align_scale: float = ALIGN_SCALE) -> float:
    """Score: how much the val ranking deserves to be trusted (0..1).

    Driven by the top-1/top-2 margin relative to the noise band.  A margin of
    three noise bands (``align_scale``) is treated as a fully trustworthy
    ordering.
    """
    if len(arms) < 2:
        return 1.0
    vals = sorted((a.val for a in arms), reverse=True)
    margin = vals[0] - vals[1]
    return float(_clip(margin / align_scale)) if align_scale else 1.0


def _clip(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, x))


# ---------------------------------------------------------------------------
# Meta-parameters (what RSI learns)
# ---------------------------------------------------------------------------
@dataclass
class MetaParams:
    collapse_threshold: float = 0.70
    alignment_threshold: float = 0.50
    confidence_floor: float = 0.60

    def clamped(self) -> "MetaParams":
        return MetaParams(
            collapse_threshold=float(np.clip(self.collapse_threshold, 0.05, 0.95)),
            alignment_threshold=float(np.clip(self.alignment_threshold, 0.05, 0.95)),
            confidence_floor=float(np.clip(self.confidence_floor, 0.05, 0.95)),
        )

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass
class Decision:
    deployed_key: str
    deployed_name: str
    reason: str
    hold: float
    regret: float
    features: Dict[str, float] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Policies
# ---------------------------------------------------------------------------
def decide_conservative(g: Group) -> Tuple[Arm, str, Dict[str, float]]:
    a = g.conservative
    return a, "always_conservative", {}


def decide_greedy(g: Group) -> Tuple[Arm, str, Dict[str, float]]:
    a = g.greedy
    return a, "greedy_val", {"margin": _margin(g), "confidence": 1.0}


def decide_heuristic(g: Group) -> Tuple[Arm, str, Dict[str, float]]:
    """Original AutoAPC-Select: val argmax, Occam tie-break inside ±noise band."""
    best = max(a.val for a in g.arms)
    tol = [a for a in g.arms if a.val >= best - NOISE_BAND]
    a = min(tol, key=lambda x: (x.rank,))
    return a, "heuristic_occam", {"margin": _margin(g), "confidence": 1.0}


class JevGate:
    """Structured gate over the three Jev primitives.

    ``enabled`` lets ablations switch individual rules off; the gate then falls
    through to the next rule exactly as if the removed primitive returned a
    permissive value.

    ``val_sigma`` / ``align_scale`` default to the frozen-corpus calibration
    (``VAL_SIGMA`` / ``ALIGN_SCALE``); passing corpus-specific values realises the
    "scale-calibrated" sensitivity analysis without touching the default
    behaviour that the frozen artifact was produced with.
    """

    ALL = ("collapse", "alignment", "confidence")

    def __init__(self, meta: Optional[MetaParams] = None,
                 enabled: Sequence[str] = ALL, fallback: str = "base",
                 val_sigma: float = VAL_SIGMA, align_scale: float = ALIGN_SCALE,
                 noise_band: float = NOISE_BAND):
        self.meta = meta or MetaParams()
        self.enabled = tuple(enabled)
        # "base"  -> fall back to the no-adaptation arm (faithful to the
        #            original hard-coded 'z0' fallback)
        # "occam" -> fall back to the family's Occam-minimum arm
        self.fallback = fallback
        self.val_sigma = val_sigma
        self.align_scale = align_scale
        self.noise_band = noise_band

    def _safe(self, g: Group) -> Arm:
        return g.base if self.fallback == "base" else g.conservative

    # -- features ---------------------------------------------------------
    @staticmethod
    def features(g: Group, val_sigma: float = VAL_SIGMA,
                 align_scale: float = ALIGN_SCALE) -> Dict[str, float]:
        probs = choice_posterior(g.arms, sigma=val_sigma)
        greedy = g.greedy
        return {
            "confidence": probs[greedy.key],
            "collapse_noul": collapse_noul(g.arms),
            "alignment": alignment_score(g.arms, align_scale=align_scale),
            "margin": _margin(g),
            "n_arms": float(g.n_arms),
        }

    # -- decision ---------------------------------------------------------
    def act(self, g: Group, feats: Optional[Dict[str, float]] = None) -> Decision:
        f = feats if feats is not None else self.features(
            g, self.val_sigma, self.align_scale)
        m = self.meta

        if "collapse" in self.enabled and f["collapse_noul"] > m.collapse_threshold:
            return self._emit(g, self._safe(g), "collapse_detected", f)

        if "alignment" in self.enabled and f["alignment"] < m.alignment_threshold:
            return self._emit(g, self._safe(g), "low_val_alignment", f)

        if "confidence" in self.enabled and f["confidence"] < m.confidence_floor:
            best = max(a.val for a in g.arms)
            tol = [a for a in g.arms if a.val >= best - self.noise_band]
            return self._emit(g, min(tol, key=lambda x: (x.rank,)),
                              "low_confidence_occam", f)

        return self._emit(g, g.greedy, "normal_selection", f)

    @staticmethod
    def _emit(g: Group, arm: Arm, reason: str, f: Dict[str, float]) -> Decision:
        return Decision(deployed_key=arm.key, deployed_name=arm.name,
                        reason=reason, hold=arm.hold,
                        regret=max(a.hold for a in g.arms) - arm.hold,
                        features=dict(f))


def _margin(g: Group) -> float:
    vals = sorted((a.val for a in g.arms), reverse=True)
    return float(vals[0] - vals[1]) if len(vals) > 1 else 0.0


# ---------------------------------------------------------------------------
# Registry used by the experiment runners
# ---------------------------------------------------------------------------
def get_policy(name: str, meta: Optional[MetaParams] = None,
               enabled: Sequence[str] = JevGate.ALL):
    if name == "conservative":
        return lambda g: _wrap(g, *decide_conservative(g))
    if name == "greedy":
        return lambda g: _wrap(g, *decide_greedy(g))
    if name == "heuristic":
        return lambda g: _wrap(g, *decide_heuristic(g))
    if name == "jev":
        gate = JevGate(meta=meta, enabled=enabled)
        return lambda g: gate.act(g)
    raise ValueError(f"unknown policy {name!r}")


def _wrap(g: Group, arm: Arm, reason: str, feats: Dict[str, float]) -> Decision:
    return Decision(deployed_key=arm.key, deployed_name=arm.name, reason=reason,
                    hold=arm.hold,
                    regret=max(a.hold for a in g.arms) - arm.hold,
                    features=dict(feats))
