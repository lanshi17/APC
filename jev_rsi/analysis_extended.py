"""Scenario-level analysis of the P3 extended corpus (plan task E2).

Produces ``results/jev_rsi_extended_analysis.json`` plus a human-readable
table:

* per-scenario decision quality for ``conservative`` / ``greedy`` / ``heuristic``
  / ``jev_static`` / ``loo_rsi`` (leave-one-group-out RSI, trained *within* the
  scenario so no scenario leaks into another);
* Scenario C: does the val-argmax get captured by a gaming arm, and does the
  gate's Score/Noul primitives notice?
* Scenario A: judge variance vs greedy regret and vs gate behaviour;
* Scenario B: per-style val/holdout means and the text-judge/vision-judge
  decoupling;
* boundary conditions: the exact groups where structured gating beats or loses
  to plain val-argmax.

Usage
-----
    python -m jev_rsi.analysis_extended
    python -m jev_rsi.analysis_extended --out jev_rsi/results
"""
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Dict, List, Optional, Sequence

from .core import JevGate, get_policy
from .data import (EXTENDED_ROOT, Group, available_scenarios, load_all_groups,
                   scenario_label, scenario_of)
from .experiments import RESULTS, Harness, _paired, loo_rsi, seen, summarize
from .metrics import spearman
from .rsi import replay

SCENARIO_ORDER = ["original", "scenario_c", "scenario_a", "scenario_b"]
SCENARIO_TITLE = {
    "original": "original frozen APC corpus (43 groups)",
    "scenario_c": "Scenario C: adversarial summarisation (10 groups)",
    "scenario_a": "Scenario A: high-variance judge (20 groups)",
    "scenario_b": "Scenario B: multi-modal alignment (15 groups)",
}


# ---------------------------------------------------------------------------
# generic per-scenario table
# ---------------------------------------------------------------------------
def scenario_sets(h: Harness) -> Dict[str, List[Group]]:
    out: Dict[str, List[Group]] = {}
    for g in h.groups:
        out.setdefault(scenario_of(g.gid), []).append(g)
    return {k: out[k] for k in SCENARIO_ORDER if k in out}


def per_scenario_table(h: Harness) -> Dict[str, dict]:
    table: Dict[str, dict] = {}
    for scen, groups in scenario_sets(h).items():
        dec = [g for g in groups if g.is_decision]
        row: Dict[str, object] = {
            "description": SCENARIO_TITLE.get(scen, scen),
            "n_groups": len(groups),
            "n_arms": sum(g.n_arms for g in groups),
            "n_decision_groups": len(dec),
            "informative_groups": [g.gid for g in groups if g.is_informative],
            "greedy_oracle_gap": round(sum(g.oracle.hold - g.greedy.hold for g in groups), 6),
        }
        for pname in ("conservative", "greedy", "heuristic"):
            row[pname] = summarize(dec, h.evaluate(pname, dec)) if dec else None
        row["jev_static"] = summarize(dec, h.evaluate("jev", dec)) if dec else None
        if len(dec) >= 3:
            row["loo_rsi"] = _scoped_loo(h, dec)
            row["paired_greedy_vs_jev_static"] = _paired(
                [d.regret for d in h.evaluate("greedy", dec)],
                [d.regret for d in h.evaluate("jev", dec)],
                "greedy", "jev_static")
        # reason distribution inside the scenario (decision groups)
        reasons: Dict[str, int] = {}
        if dec:
            gate = JevGate()
            for g in dec:
                r = gate.act(g, h.feats[g.gid]).reason
                reasons[r] = reasons.get(r, 0) + 1
        row["gate_reasons"] = reasons
        table[scen] = row
    return table


def _scoped_loo(h: Harness, groups: Sequence[Group],
                gate_kwargs: Optional[dict] = None) -> dict:
    """LOO-RSI computed only over the given (scenario-local) decision groups."""
    gate_kwargs = gate_kwargs or {}
    ds = []
    for i, test in enumerate(groups):
        train = [g for j, g in enumerate(groups) if j != i]
        learner = replay(train, clustering=True)
        gate = JevGate(meta=learner.meta, **gate_kwargs)
        ds.append(gate.act(test, h.feats[test.gid]))
    return summarize(groups, ds)


# ---------------------------------------------------------------------------
# scale calibration sensitivity
# ---------------------------------------------------------------------------
def fit_spacing_scale(groups: Sequence[Group]) -> dict:
    """Estimate a val-score spacing scale from the corpus itself.

    Median absolute gap between adjacent arms inside a group -- a robust proxy
    for "one step of val noise" that does not need repeated measurements.  The
    aligned scale is 3x that, mirroring ``ALIGN_SCALE = 3 * NOISE_BAND``.
    """
    gaps: List[float] = []
    for g in groups:
        vals = sorted((a.val for a in g.arms), reverse=True)
        gaps.extend(vals[i] - vals[i + 1] for i in range(len(vals) - 1))
    if not gaps:
        return {"sigma": 0.0, "align_scale": 0.0, "noise_band": 0.0, "n_gaps": 0}
    sigma = statistics.median(gaps)
    return {"sigma": round(sigma, 6), "align_scale": round(3 * sigma, 6),
            "noise_band": round(sigma, 6), "n_gaps": len(gaps)}


def scale_sensitivity(h: Harness) -> dict:
    """Re-run each scenario with its *own* val scale instead of the frozen one."""
    out: Dict[str, dict] = {}
    for scen, groups in scenario_sets(h).items():
        dec = [g for g in groups if g.is_decision]
        if len(dec) < 2:
            continue
        scale = fit_spacing_scale(dec)
        if not scale["sigma"]:
            continue
        kwargs = {"val_sigma": scale["sigma"], "align_scale": scale["align_scale"],
                  "noise_band": scale["noise_band"]}
        feats = {g.gid: JevGate.features(g, scale["sigma"], scale["align_scale"])
                 for g in dec}
        gate = JevGate(**kwargs)
        ds = [gate.act(g, feats[g.gid]) for g in dec]
        row = {
            "scale": scale,
            "fraction_alignment_saturated": round(
                sum(1 for f in feats.values() if f["alignment"] >= 1.0) / len(feats), 3),
            "greedy": summarize(dec, h.evaluate("greedy", dec)),
            "jev_static_calibrated": summarize(dec, ds),
            "gate_reasons": {},
        }
        for d in ds:
            row["gate_reasons"][d.reason] = row["gate_reasons"].get(d.reason, 0) + 1
        if len(dec) >= 3:
            row["loo_rsi_calibrated"] = _scoped_loo(h, dec, gate_kwargs=kwargs)
        out[scen] = row
    return out


# ---------------------------------------------------------------------------
# Scenario C -- adversarial decoupling
# ---------------------------------------------------------------------------
def _scores_json(name: str) -> Optional[dict]:
    path = EXTENDED_ROOT / name / "scores.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def scenario_c_analysis(h: Harness) -> dict:
    data = _scores_json("scenario_c")
    if not data:
        return {"available": False}
    rows = []
    n_capture = 0
    for gid, rec in sorted(data["groups"].items()):
        arms = rec["arms"]
        vals = [a["val_score"] for a in arms]
        holds = [a["holdout_score"] for a in arms]
        greedy = max(arms, key=lambda a: a["val_score"])
        best = max(arms, key=lambda a: a["holdout_score"])
        normal = [a for a in arms if a["type"] == "normal"]
        gaming = [a for a in arms if a["type"] == "gaming"]
        captured = greedy["type"] == "gaming" and best["type"] != "gaming"
        n_capture += int(captured)
        group = next(g for g in h.groups if g.gid == gid)
        gate = JevGate().act(group, h.feats[gid])
        f = h.feats[gid]
        rows.append({
            "gid": gid,
            "val_holdout_spearman": round(spearman(vals, holds), 4),
            "greedy_arm": greedy["id"], "greedy_type": greedy["type"],
            "oracle_arm": best["id"], "oracle_type": best["type"],
            "captured_by_gaming": captured,
            "gaming_val_max": round(max(a["val_score"] for a in gaming), 4),
            "normal_holdout_max": round(max(a["holdout_score"] for a in normal), 4),
            "gaming_holdout_max": round(max(a["holdout_score"] for a in gaming), 4),
            "gate_arm": gate.deployed_name, "gate_reason": gate.reason,
            "gate_regret": round(gate.regret, 6),
            "greedy_regret": round(max(a["holdout_score"] for a in arms)
                                   - greedy["holdout_score"], 6),
            "features": {k: round(v, 4) for k, v in f.items()},
        })
    per_arm: Dict[str, dict] = {}
    for rec in data["groups"].values():
        for a in rec["arms"]:
            e = per_arm.setdefault(a["id"], {"type": a["type"], "val": [], "hold": []})
            e["val"].append(a["val_score"])
            e["hold"].append(a["holdout_score"])
    arms_summary = {k: {"type": v["type"],
                        "mean_val": round(statistics.fmean(v["val"]), 4),
                        "mean_holdout": round(statistics.fmean(v["hold"]), 4)}
                    for k, v in per_arm.items()}
    by_type: Dict[str, dict] = {}
    for k, v in arms_summary.items():
        e = by_type.setdefault(v["type"], {"mean_val": [], "mean_holdout": []})
        e["mean_val"].append(v["mean_val"])
        e["mean_holdout"].append(v["mean_holdout"])
    type_summary = {t: {"mean_val": round(statistics.fmean(v["mean_val"]), 4),
                        "mean_holdout": round(statistics.fmean(v["mean_holdout"]), 4),
                        "n_arms": len(v["mean_val"])}
                    for t, v in by_type.items()}
    # which recorded val metric (if any) would have been captured by a gaming arm?
    metric_names = sorted({m for rec in data["groups"].values()
                           for a in rec["arms"] for m in a["metrics"]
                           if m.startswith(("rouge", "bertscore"))})
    variant_capture = {}
    for m in metric_names:
        n = 0
        for rec in data["groups"].values():
            top = max(rec["arms"], key=lambda a: a["metrics"][m])
            if top["type"] == "gaming":
                n += 1
        variant_capture[m] = n
    all_vals = [a["val_score"] for rec in data["groups"].values() for a in rec["arms"]]
    return {
        "available": True,
        "protocol": data.get("protocol"),
        "bertscore_model": data.get("bertscore_model"),
        "n_groups_captured_by_gaming": n_capture,
        "n_groups": len(rows),
        "mean_val_holdout_spearman": round(
            statistics.fmean(r["val_holdout_spearman"] for r in rows), 4),
        "val_min": round(min(all_vals), 4),
        "val_max": round(max(all_vals), 4),
        "arms": arms_summary,
        "by_type": type_summary,
        "val_variant_capture": variant_capture,
        "rows": rows,
    }


# ---------------------------------------------------------------------------
# Scenario A -- judge variance
# ---------------------------------------------------------------------------
def scenario_a_analysis(h: Harness) -> dict:
    data = _scores_json("scenario_a")
    if not data:
        return {"available": False}
    rows = []
    for gid, rec in sorted(data["groups"].items()):
        group = next(g for g in h.groups if g.gid == gid)
        gate = JevGate().act(group, h.feats[gid])
        greedy = group.greedy
        rows.append({
            "gid": gid,
            "kind": rec.get("kind"),
            "judge_std_mean_10": rec["judge_variance_10"]["mean_arm_std"],
            "judge_std_max_10": rec["judge_variance_10"]["max_arm_std"],
            "greedy_arm": greedy.name, "oracle_arm": group.oracle.name,
            "greedy_regret": round(group.oracle.hold - greedy.hold, 6),
            "gate_arm": gate.deployed_name, "gate_reason": gate.reason,
            "gate_regret": round(gate.regret, 6),
            "features": {k: round(v, 4) for k, v in h.feats[gid].items()},
        })
    stds = [r["judge_std_mean_10"] for r in rows]
    high = [r for r in rows if r["judge_std_mean_10"] > 2.0]
    retreat = [r for r in rows if r["gate_reason"] != "normal_selection"]
    return {
        "available": True,
        "judge_model": (data.get("provenance") or {}).get("judge_model"),
        "n_groups": len(rows),
        "n_high_variance_groups": len(high),
        "mean_judge_std_10": round(statistics.fmean(stds), 4),
        "spearman_variance_vs_greedy_regret": round(
            spearman(stds, [r["greedy_regret"] for r in rows]), 4),
        "n_gate_retreats": len(retreat),
        "rows": rows,
    }


# ---------------------------------------------------------------------------
# Scenario B -- multi-modal alignment
# ---------------------------------------------------------------------------
def scenario_b_analysis(h: Harness) -> dict:
    data = _scores_json("scenario_b")
    if not data:
        return {"available": False}
    rows = []
    n_capture = 0
    per_style: Dict[str, dict] = {}
    for gid, rec in sorted(data["groups"].items()):
        arms = rec["arms"]
        greedy = max(arms, key=lambda a: a["val_score"])
        best = max(arms, key=lambda a: a["holdout_score"])
        captured = greedy["id"] in ("poetic", "verbose", "child_like") and \
            best["id"] in ("factual", "technical", "concise")
        n_capture += int(captured)
        group = next(g for g in h.groups if g.gid == gid)
        gate = JevGate().act(group, h.feats[gid])
        rows.append({
            "gid": gid,
            "val_holdout_spearman": round(spearman([a["val_score"] for a in arms],
                                                   [a["holdout_score"] for a in arms]), 4),
            "greedy_style": greedy["id"], "oracle_style": best["id"],
            "captured_by_poetic": captured,
            "greedy_regret": round(max(a["holdout_score"] for a in arms)
                                   - greedy["holdout_score"], 6),
            "gate_arm": gate.deployed_name, "gate_reason": gate.reason,
            "gate_regret": round(gate.regret, 6),
        })
        for a in arms:
            e = per_style.setdefault(a["id"], {"val": [], "hold": []})
            e["val"].append(a["val_score"])
            e["hold"].append(a["holdout_score"])
    style_summary = {k: {"mean_val": round(statistics.fmean(v["val"]), 4),
                         "mean_holdout": round(statistics.fmean(v["hold"]), 4)}
                     for k, v in per_style.items()}
    has_rank = all("holdout_rank_score" in a
                   for rec in data["groups"].values() for a in rec["arms"])
    rank_summary = {}
    if has_rank:
        per_rank: Dict[str, List[float]] = {}
        for rec in data["groups"].values():
            for a in rec["arms"]:
                per_rank.setdefault(a["id"], []).append(a["holdout_rank_score"])
        rank_summary = {"mean_rank_score": {k: round(statistics.fmean(v), 4)
                                           for k, v in per_rank.items()},
                        "mean_spearman_val_rank": round(statistics.fmean(
                            rec["val_vs_rank_spearman"]
                            for rec in data["groups"].values()), 6),
                        "n_groups_negative": sum(
                            1 for rec in data["groups"].values()
                            if rec["val_vs_rank_spearman"] < 0)}
        for r in rows:
            rec = data["groups"][r["gid"]]
            r["val_vs_rank_spearman"] = rec["val_vs_rank_spearman"]
            best_rank = max(rec["arms"], key=lambda a: a["holdout_rank_score"])
            r["rank_oracle_style"] = best_rank["id"]
            r["greedy_rank_score"] = next(
                a["holdout_rank_score"] for a in rec["arms"]
                if a["id"] == r["greedy_style"])
            r["rank_regret"] = round(max(a["holdout_rank_score"] for a in rec["arms"])
                                     - r["greedy_rank_score"], 6)
    return {
        "available": True,
        "val_judge": (data.get("provenance") or {}).get("val_judge"),
        "holdout_judge": (data.get("provenance") or {}).get("holdout_judge"),
        "n_groups": len(rows),
        "n_groups_captured_by_poetic": n_capture,
        "styles": style_summary,
        "ranking_holdout": rank_summary,
        "mean_val_holdout_spearman": round(
            statistics.fmean(r["val_holdout_spearman"] for r in rows), 6),
        "rows": rows,
    }


# ---------------------------------------------------------------------------
# boundary conditions
# ---------------------------------------------------------------------------
def boundary_conditions(h: Harness) -> dict:
    wins, losses, ties = [], [], []
    for g in h.decision_groups:
        greedy = get_policy("greedy")(g)
        jev = JevGate().act(g, h.feats[g.gid])
        row = {"gid": g.gid, "scenario": scenario_of(g.gid),
               "greedy_arm": greedy.deployed_name, "jev_arm": jev.deployed_name,
               "jev_reason": jev.reason,
               "greedy_regret": round(greedy.regret, 6),
               "jev_regret": round(jev.regret, 6),
               "delta": round(greedy.regret - jev.regret, 6)}
        (wins if row["delta"] > 1e-9 else losses if row["delta"] < -1e-9 else ties).append(row)
    by_scen: Dict[str, dict] = {}
    for scen in SCENARIO_ORDER:
        w = [r for r in wins if r["scenario"] == scen]
        l = [r for r in losses if r["scenario"] == scen]
        t = [r for r in ties if r["scenario"] == scen]
        if w or l or t:
            by_scen[scen] = {"wins": len(w), "losses": len(l), "ties": len(t),
                             "win_ids": [r["gid"] for r in w],
                             "loss_ids": [r["gid"] for r in l]}
    return {"wins": wins, "losses": losses, "ties": ties, "by_scenario": by_scen}


def fmt_report(payload: dict) -> str:
    L = ["=== P3 EXTENDED CORPUS: SCENARIO ANALYSIS ===", ""]
    ds = payload["dataset"]
    L.append(f"groups={ds['n_total']} (frozen {ds['n_frozen']} + extended {ds['n_extended']}) "
             f"arms={ds['n_arms_total']} decision={ds['n_decision_total']}")
    L.append("")
    L.append("--- per-scenario decision quality (decision groups) ---")
    for scen, row in payload["per_scenario"].items():
        if not row.get("n_decision_groups"):
            continue
        L.append(f"{scen:11s} n={row['n_decision_groups']:<3d}")
        for p in ("conservative", "greedy", "heuristic", "jev_static"):
            if row.get(p):
                L.append(f"   {p:13s} {seen(row[p])}")
        if row.get("loo_rsi"):
            L.append(f"   {'loo_rsi':13s} {seen(row['loo_rsi'])}")
        if row.get("paired_greedy_vs_jev_static"):
            v = row["paired_greedy_vs_jev_static"]
            L.append(f"   greedy-vs-jev W/L/T={v['wins']}/{v['losses']}/{v['ties']} "
                     f"mean_diff={v['mean_diff']:+.5f} p={v['sign_test_p']}")
        L.append(f"   gate reasons: {row['gate_reasons']}")
    bc = payload["boundary_conditions"]
    L.append("")
    L.append("--- boundary conditions (structured gate vs val-argmax) ---")
    for scen, row in bc["by_scenario"].items():
        L.append(f"{scen:11s} wins={row['wins']} losses={row['losses']} ties={row['ties']}"
                 f"  win_ids={row['win_ids']}")
    c = payload["scenario_c"]
    if c.get("available"):
        L.append("")
        L.append(f"--- Scenario C --- captured by a gaming arm: "
                 f"{c['n_groups_captured_by_gaming']}/{c['n_groups']}; "
                 f"mean Spearman(val,holdout)={c['mean_val_holdout_spearman']}")
        L.append(f"   arms: " + "  ".join(
            f"{k}({v['type']}) val={v['mean_val']:.3f}/hold={v['mean_holdout']:.3f}"
            for k, v in c["arms"].items()))
    a = payload["scenario_a"]
    if a.get("available"):
        L.append("")
        L.append(f"--- Scenario A --- high-variance groups: "
                 f"{a['n_high_variance_groups']}/{a['n_groups']}; "
                 f"mean judge std={a['mean_judge_std_10']}; "
                 f"Spearman(variance,greedy regret)={a['spearman_variance_vs_greedy_regret']}; "
                 f"gate retreats={a['n_gate_retreats']}")
    b = payload["scenario_b"]
    if b.get("available"):
        L.append("")
        L.append(f"--- Scenario B --- captured by poetic/verbose: "
                 f"{b['n_groups_captured_by_poetic']}/{b['n_groups']}; "
                 f"mean Spearman(val,holdout)={b['mean_val_holdout_spearman']}")
        L.append("   styles: " + "  ".join(
            f"{k} val={v['mean_val']:.3f}/hold={v['mean_holdout']:.3f}"
            for k, v in b["styles"].items()))
        rk = b.get("ranking_holdout") or {}
        if rk:
            L.append(f"   ranking holdout: mean Spearman(val,rank)="
                     f"{rk['mean_spearman_val_rank']:+.3f} "
                     f"(negative in {rk['n_groups_negative']}/{b['n_groups']} groups); "
                     f"per-style " + "  ".join(
                         f"{k}={v:.3f}" for k, v in rk["mean_rank_score"].items()))
    L.append("")
    L.append("--- calibration (gate noise band vs new val scales) ---")
    for scen, c in payload["calibration"].items():
        L.append(f"{scen:11s} val range [{c['val_min']}, {c['val_max']}] "
                 f"median margin={c['median_margin']} "
                 f"(= {c['margin_over_noise_band']}x NOISE_BAND); "
                 f"alignment saturated in {c['frac_alignment_saturated']:.0%} of groups; "
                 f"noul==0 in {c['frac_collapse_noul_zero']:.0%}")
    if payload.get("scale_sensitivity"):
        L.append("")
        L.append("--- scale-calibrated sensitivity (gate refit to each scenario's "
                 "own val spacing) ---")
        for scen, row in payload["scale_sensitivity"].items():
            L.append(f"{scen:11s} sigma={row['scale']['sigma']:.4f} "
                     f"align_scale={row['scale']['align_scale']:.4f} "
                     f"(alignment saturated {row['fraction_alignment_saturated']:.0%})")
            L.append(f"   greedy      {seen(row['greedy'])}")
            L.append(f"   jev_static  {seen(row['jev_static_calibrated'])}  "
                     f"reasons={row['gate_reasons']}")
            if row.get("loo_rsi_calibrated"):
                L.append(f"   loo_rsi     {seen(row['loo_rsi_calibrated'])}")
    jv = payload.get("judge_variance_probe") or {}
    if jv.get("available"):
        L.append("")
        L.append("--- judge-variance probe (can the high-variance regime exist?) ---")
        for model, rec in jv["models"].items():
            L.append(f"{model:14s} {rec['draws']} draws x {rec['n_arms']} arms: "
                     f"mean sd={rec['mean_arm_std_10']:.3f} max sd={rec['max_arm_std_10']:.3f} "
                     f"(0-10); arms with sd>2.0: {rec['n_arms_std_gt_2']}/{rec['n_arms']}")
    return "\n".join(L)


# ---------------------------------------------------------------------------
# calibration diagnostic
# ---------------------------------------------------------------------------
def calibration_diagnostic(h: Harness) -> dict:
    """Is the gate's scale still appropriate for the new val metrics?

    ``core.NOISE_BAND`` (0.007) and ``ALIGN_SCALE`` (0.021) were fitted on the
    frozen APC corpus, whose ``val_score`` is an 8-point benchmark fraction.
    The extended scenarios use ROUGE-L F1 and 0-1-normalised LLM-judge scores,
    whose natural noise and spread are much larger.  This diagnostic reports how
    often each rule is saturated, without changing the frozen default behaviour.
    """
    from .core import ALIGN_SCALE, NOISE_BAND
    out: Dict[str, dict] = {}
    for scen, groups in scenario_sets(h).items():
        dec = [g for g in groups if g.is_decision]
        if not dec:
            continue
        feats = [h.feats[g.gid] for g in dec]
        vals = [a.val for g in dec for a in g.arms]
        margins = [f["margin"] for f in feats]
        out[scen] = {
            "n_decision_groups": len(dec),
            "val_min": round(min(vals), 4),
            "val_median": round(statistics.median(vals), 4),
            "val_max": round(max(vals), 4),
            "median_margin": round(statistics.median(margins), 4),
            "margin_over_noise_band": round(
                statistics.median(margins) / NOISE_BAND, 1),
            "frac_alignment_saturated": round(
                sum(1 for f in feats if f["alignment"] >= 1.0) / len(feats), 3),
            "frac_collapse_noul_zero": round(
                sum(1 for f in feats if f["collapse_noul"] <= 1e-9) / len(feats), 3),
            "noise_band": NOISE_BAND,
            "align_scale": ALIGN_SCALE,
        }
    return out


def judge_variance_probe() -> dict:
    """Read the cross-model judge-variance probe artifact, if it was run."""
    path = EXTENDED_ROOT / "scenario_a" / "judge_variance_probe.json"
    if not path.exists():
        return {"available": False}
    data = json.loads(path.read_text(encoding="utf-8"))
    return {"available": True,
            "protocol": data.get("protocol"),
            "models": {m: {k: v for k, v in rec.items() if k != "per_arm"}
                       for m, rec in data["models"].items()}}


def run_all(outdir: Path = RESULTS) -> dict:
    h = Harness(load_all_groups())
    from .data import corpus_audit
    payload = {
        "dataset": corpus_audit(),
        "per_scenario": per_scenario_table(h),
        "boundary_conditions": boundary_conditions(h),
        "scenario_c": scenario_c_analysis(h),
        "scenario_a": scenario_a_analysis(h),
        "scenario_b": scenario_b_analysis(h),
        "calibration": calibration_diagnostic(h),
        "scale_sensitivity": scale_sensitivity(h),
        "judge_variance_probe": judge_variance_probe(),
    }
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "jev_rsi_extended_analysis.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return payload


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", default=str(RESULTS))
    args = ap.parse_args(argv)
    payload = run_all(Path(args.out))
    print(fmt_report(payload))
    print(f"\nwrote {Path(args.out)/'jev_rsi_extended_analysis.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
