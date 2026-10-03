"""Real-Jev replication of the F12 gate over the frozen corpus.

The offline harness scores the gate with *mock* Jev primitives computed
in ``jev_rsi/core.py``.  This module re-runs the exact same protocol --
same groups, same three gate rules, the same ``summarize`` metrics, the
same ``NOISE_BAND`` -- with the three features supplied by a **live**
``POST /v1/systemone`` call instead:

    confidence     <- answers.best_arm.confidence
    collapse_noul  <- answers.seed_collapse.noul
    alignment      <- answers.signal_quality.score / (len(legend) - 1)

Nothing else changes, so any difference between this table and the
offline one is attributable to the feature source alone.

Usage
-----
    python -m jev_rsi.real_jev                       # all decision groups
    python -m jev_rsi.real_jev --limit 3             # smoke test
    python -m jev_rsi.real_jev --models jev-1.13     # pin the model
    python -m jev_rsi.real_jev --out /tmp/x          # write elsewhere

Credentials come from ``<code-repo>/.env`` (``JEV_API_BASE``,
``JEV_API_KEY``, ``JEV_MODEL``); the key is never written to the
artifact.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from .core import NOISE_BAND, Decision, JevGate, MetaParams, get_policy
from .data import APC_ROOT, Group, load_groups
from .experiments import seen, summarize

DEFAULT_OUT = Path(__file__).resolve().parent / "results"
COST_PER_M_INPUT = 0.042  # USD / 1M input tokens (TypeSafe list price)


# ---------------------------------------------------------------------------
# credential + client plumbing
# ---------------------------------------------------------------------------
def _load_env() -> Dict[str, str]:
    env: Dict[str, str] = {}
    dotenv = APC_ROOT / ".env"
    if dotenv.exists():
        for line in dotenv.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                env[k.strip()] = v.strip()
    return env


def _client_factory():
    """Import the real client from the sister pipeline package."""
    pipeline = APC_ROOT / "apc-pipeline"
    if str(pipeline) not in sys.path:
        sys.path.insert(0, str(pipeline))
    from apc.decision.jev_client_real import JevClient  # noqa: E402

    return JevClient


def _seed_of(g: Group) -> int:
    m = re.search(r"-s(\d+)$", g.gid)
    return int(m.group(1)) if m else 0


# ---------------------------------------------------------------------------
# the gate, driven by externally supplied features
# ---------------------------------------------------------------------------
def gate_from_features(g: Group, f: Dict[str, float],
                       meta: MetaParams, fallback: str = "base") -> Decision:
    """Byte-for-byte the rule order of ``JevGate.act``."""
    safe = g.base if fallback == "base" else g.conservative

    if f["collapse_noul"] > meta.collapse_threshold:
        arm, reason = safe, "collapse_detected"
    elif f["alignment"] < meta.alignment_threshold:
        arm, reason = safe, "low_val_alignment"
    elif f["confidence"] < meta.confidence_floor:
        best = max(a.val for a in g.arms)
        tol = [a for a in g.arms if a.val >= best - NOISE_BAND]
        arm, reason = min(tol, key=lambda x: x.rank), "low_confidence_occam"
    else:
        arm, reason = g.greedy, "normal_selection"

    return Decision(deployed_key=arm.key, deployed_name=arm.name, reason=reason,
                    hold=arm.hold,
                    regret=max(a.hold for a in g.arms) - arm.hold,
                    features=dict(f))


def _arms_payload(g: Group) -> List[Dict[str, Any]]:
    return [{"name": a.name, "val_score": a.val, "rank": a.rank} for a in g.arms]


# ---------------------------------------------------------------------------
# one live pass
# ---------------------------------------------------------------------------
def run_model(model: str, groups: Sequence[Group], client,
              meta: MetaParams, retries: int = 3) -> Dict[str, Any]:
    rows: List[Dict[str, Any]] = []
    decisions: List[Decision] = []
    tok_in = tok_out = 0
    n_live = n_fallback = 0

    for i, g in enumerate(groups, 1):
        arms = _arms_payload(g)
        ctx = {"task": g.task, "seed": _seed_of(g)}
        out: Optional[Dict[str, Any]] = None
        err = None
        t0 = time.time()
        for attempt in range(retries):
            try:
                out = client.select_best_arm(arms, ctx)
                break
            except Exception as e:  # noqa: BLE001
                err = f"{type(e).__name__}: {str(e)[:180]}"
                time.sleep(2.0 * (attempt + 1))
        latency = round(time.time() - t0, 3)

        if out is None:
            print(f"  [{i:>2}/{len(groups)}] {g.gid}: FAILED after {retries} tries ({err})")
            continue

        live = out.get("method") == "jev"
        n_live += int(live)
        n_fallback += int(not live)

        usage = out.get("usage") or {}
        tok_in += int(usage.get("input_tokens", 0))
        tok_out += int(usage.get("output_tokens", 0))

        feats = {
            "confidence": float(out.get("confidence", 0.0)),
            "collapse_noul": float(out.get("collapse_noul", 0.0)),
            "alignment": float(out.get("alignment_score", 0.0)),
            "margin": (max(a.val for a in g.arms) - sorted((a.val for a in g.arms), reverse=True)[1]
                       if len(g.arms) > 1 else 0.0),
            "n_arms": float(g.n_arms),
        }
        dec = gate_from_features(g, feats, meta)
        decisions.append(dec)

        rows.append({
            "gid": g.gid, "task": g.task, "n_arms": g.n_arms,
            "oracle": g.oracle.name, "greedy": g.greedy.name,
            "val_scores": {a.name: round(a.val, 4) for a in g.arms},
            "holdout_scores": {a.name: round(a.hold, 4) for a in g.arms},
            "live": live, "method": out.get("method"), "model": out.get("model"),
            "jev_raw_choice": out.get("choice"),
            "jev_raw_correct": out.get("choice") == g.oracle.name,
            "probabilities": {k: round(v, 4) for k, v in (out.get("probabilities") or {}).items()},
            "features": {k: round(v, 4) for k, v in feats.items()},
            "gate_choice": dec.deployed_name, "gate_reason": dec.reason,
            "gate_correct": dec.regret < 1e-9,
            "regret": round(dec.regret, 6),
            "usage": usage, "latency_s": latency,
        })
        mark = "OK " if dec.regret < 1e-9 else "   "
        print(f"  [{i:>2}/{len(groups)}] {g.gid:<18} {mark} raw={out.get('choice'):<14} "
              f"gate={dec.deployed_name:<14} {dec.reason:<20} "
              f"oracle={g.oracle.name:<14} live={live} {latency:.1f}s")

    summary = summarize(groups[:len(decisions)], decisions)
    raw_exact = sum(1 for r in rows if r["jev_raw_correct"])
    return {
        "model_requested": model,
        "models_returned": sorted({r["model"] for r in rows if r.get("model")}),
        "live_calls": n_live,
        "fallback_calls": n_fallback,
        "usage": {"input_tokens": tok_in, "output_tokens": tok_out,
                  "est_cost_usd": round(tok_in / 1e6 * COST_PER_M_INPUT, 6)},
        "raw_choice_exact_oracle": raw_exact,
        "raw_choice_exact_rate": round(raw_exact / len(rows), 4) if rows else 0.0,
        "summary": summary,
        "groups": rows,
    }


# ---------------------------------------------------------------------------
# entry point
# ---------------------------------------------------------------------------
def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--models", nargs="*", default=None,
                    help="Jev model ids (default: JEV_MODEL from .env)")
    ap.add_argument("--limit", type=int, default=0, help="only the first N groups")
    ap.add_argument("--manifest", default=None, help="write here instead of results/")
    args = ap.parse_args(argv)

    env = _load_env()
    if not env.get("JEV_API_BASE") or not env.get("JEV_API_KEY"):
        print("ERROR: JEV_API_BASE / JEV_API_KEY missing from .env", file=sys.stderr)
        return 2
    models = args.models or [env.get("JEV_MODEL", "jev-1.13")]

    JevClient = _client_factory()
    meta = MetaParams()
    groups = [g for g in load_groups() if g.is_decision]
    if args.limit:
        groups = groups[:args.limit]

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"endpoint : {env['JEV_API_BASE']}/systemone")
    print(f"groups   : {len(groups)} decision groups")
    print(f"models   : {', '.join(models)}")
    print(f"meta     : collapse>{meta.collapse_threshold}  "
          f"align<{meta.alignment_threshold}  conf<{meta.confidence_floor}\n")

    artifact: Dict[str, Any] = {
        "protocol": ("F12 gate over frozen corpus, three features supplied by a live "
                     "POST /v1/systemone; rules and metrics identical to the offline run"),
        "endpoint": f"{env['JEV_API_BASE'].rstrip('/')}/systemone",
        "noise_band": NOISE_BAND,
        "meta_params": {"collapse_threshold": meta.collapse_threshold,
                        "alignment_threshold": meta.alignment_threshold,
                        "confidence_floor": meta.confidence_floor},
        "n_decision_groups": len(groups),
        "models": {},
    }

    # offline baselines on the same group set
    baselines = {}
    for pname in ("greedy", "heuristic"):
        fn = get_policy(pname)
        baselines[pname] = summarize(groups, [fn(g) for g in groups])
    mock_gate = JevGate(meta=meta)
    baselines["jev_mock"] = summarize(groups, [mock_gate.act(g) for g in groups])
    artifact["baselines"] = baselines

    clients = {}
    for model in models:
        print(f"=== live pass: {model} ===")
        if model not in clients:
            clients[model] = JevClient(api_base=env["JEV_API_BASE"],
                                       api_key=env["JEV_API_KEY"],
                                       model=model, timeout=90)
        res = run_model(model, groups, clients[model], meta)
        artifact["models"][model] = res
        print(f"  -> live {res['live_calls']}/{len(groups)}  "
              f"raw-oracle {res['raw_choice_exact_oracle']}/{len(groups)}  "
              f"gated {seen(res['summary'])}")
        print(f"  -> tokens in={res['usage']['input_tokens']} "
              f"out={res['usage']['output_tokens']} "
              f"est=${res['usage']['est_cost_usd']}\n")

    print("=== comparison (decision groups, n=%d) ===" % len(groups))
    for name, s in artifact["baselines"].items():
        print(f"  {name:<12} {seen(s)}")
    for model, res in artifact["models"].items():
        print(f"  {model:<12} {seen(res['summary'])}   [live]")

    path = Path(args.manifest) if args.manifest else out_dir / "real_jev_results.json"
    path.write_text(json.dumps(artifact, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nwrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
