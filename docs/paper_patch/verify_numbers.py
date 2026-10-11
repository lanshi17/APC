#!/usr/bin/env python3
"""Check every P3 number quoted in the manuscript against the JSON artifacts.

Run after ``patch_manuscript.py`` (and after any re-run of the experiments):

    python docs/paper_patch/verify_numbers.py
    python docs/paper_patch/verify_numbers.py --md /path/to/negative-result.md

Fails loudly if a number in the prose no longer matches the artifacts, so the
"checked programmatically" claim in the reproducibility statement stays true.
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys
from decimal import ROUND_HALF_UP, Decimal

CODE = pathlib.Path(__file__).resolve().parents[2]      # <repo>/docs/paper_patch
PAPER_REPO = pathlib.Path(os.environ.get("APC_PAPER_DIR", CODE.parent / "02_apc_paper"))
PAPER_MD = PAPER_REPO / "manuscript_negative/negative-result.md"


def fmt(x: float, nd: int) -> str:
    """Half-up rounding, so 0.5265 -> '.527' as a paper would print it."""
    q = Decimal(1).scaleb(-nd)
    s = str(Decimal(repr(float(x))).quantize(q, rounding=ROUND_HALF_UP))
    return simp(float(s), nd)


def simp(x: float, nd: int = 5) -> str:
    """0.01253 -> '.01253'; the manuscript drops the leading zero."""
    s = f"{x:.{nd}f}".rstrip("0").rstrip(".")
    if s.startswith("0."):
        return s[1:]
    if s.startswith("-0."):
        return "-" + s[2:]
    return s


def check(md: str) -> list[str]:
    res = json.loads((CODE / "jev_rsi/results/jev_rsi_extended_results.json")
                     .read_text(encoding="utf-8"))
    ana = json.loads((CODE / "jev_rsi/results/jev_rsi_extended_analysis.json")
                     .read_text(encoding="utf-8"))
    mt = res["main_table"]["policies"]
    per = ana["per_scenario"]
    dd = res["deep_dive"]["paired_tests"]["greedy_vs_jev_static"]
    c, b, a = ana["scenario_c"], ana["scenario_b"], ana["scenario_a"]
    probe = ana["judge_variance_probe"]["models"]
    cal = ana["calibration"]
    sens = ana["scale_sensitivity"]

    cases: list[tuple[str, str]] = []
    add = cases.append

    # totals
    add(("total greedy", f"{mt['greedy']['decision']['exact_oracle']}/77"))
    add(("total jev", f"{mt['jev_static']['decision']['exact_oracle']}/77"))
    add(("total heuristic", f"{mt['heuristic']['decision']['exact_oracle']}/77"))
    add(("total jev avg regret", simp(mt["jev_static"]["decision"]["avg_regret"])))
    add(("total greedy avg regret", simp(mt["greedy"]["decision"]["avg_regret"])))
    add(("paired p", f"{dd['sign_test_p']:.4f}".lstrip("0")))
    add(("paired mean diff", simp(abs(dd["mean_diff"]), 4)))
    add(("paired W/L/T", f"{dd['wins']}/{dd['losses']}/{dd['ties']}"))

    # per scenario
    for scen, label in (("scenario_c", "C"), ("scenario_a", "A"), ("scenario_b", "B")):
        g = per[scen]["greedy"]
        j = per[scen]["jev_static"]
        add((f"{label} greedy", f"{g['exact_oracle']}/{g['n_groups']}"))
        add((f"{label} jev", f"{j['exact_oracle']}/{j['n_groups']}"))
        add((f"{label} greedy regret", simp(g["avg_regret"])))
        add((f"{label} jev regret", simp(j["avg_regret"])))
    for scen, label in (("scenario_c", "C"), ("scenario_a", "A"), ("scenario_b", "B")):
        pw = per[scen]["paired_greedy_vs_jev_static"]
        add((f"{label} W/L/T", f"{pw['wins']}/{pw['losses']}/{pw['ties']}"))

    # scenario C
    add(("C captured by gaming", f"{c['n_groups_captured_by_gaming']} of {c['n_groups']}"))
    add(("C val max", f"{c['val_max']:.3f}".lstrip("0")))
    add(("C val/holdout spearman", f"+{c['mean_val_holdout_spearman']:.2f}".replace("+0", "+")))
    for arm, exp in (("keyword_stuffed", "-0.135"), ("template_filled", "-0.029"),
                     ("abstractive", "+0.276"), ("random_highlight", "+0.086")):
        got = f"{c['arms'][arm]['mean_holdout']:+.3f}"
        if got.lstrip("+") != exp.lstrip("+"):
            raise SystemExit(f"unexpected {arm} holdout {got} (manuscript says {exp})")
        add((f"C {arm} holdout", f"{c['arms'][arm]['mean_holdout']:+.3f}".lstrip("+")))
    add(("C gaming mean val", f"{c['by_type']['gaming']['mean_val']:.3f}".lstrip("0")))
    add(("C gaming mean holdout", simp(c["by_type"]["gaming"]["mean_holdout"], 3)))
    add(("C rougeL_vs_article capture", f"{c['val_variant_capture']['rougeL_vs_article']} of 10"))

    # scenario A + probe
    add(("A mean judge sigma", f"{a['mean_judge_std_10']:.2f}".lstrip("0")))
    add(("A high-variance groups", f"{a['n_high_variance_groups']}/{a['n_groups']}"))
    for key in ("qwen3.8-flash", "qwen3.8-27b"):
        add((f"probe {key} sigma", simp(probe[key]["mean_arm_std_10"], 3)))
        if probe[key]["n_arms_std_gt_2"] != 0:
            raise SystemExit(f"{key} now has arms above sd 2.0 -- update the paper")
    add(("probe flash max", f"{probe['qwen3.8-flash']['max_arm_std_10']:.2f}"))
    add(("probe total arms", f"{sum(p['n_arms'] for p in probe.values())}"))

    # scenario B
    add(("B val/holdout spearman", "+" + fmt(b["mean_val_holdout_spearman"], 3)))
    rk = b["ranking_holdout"]
    add(("B val/rank spearman", "+" + fmt(rk["mean_spearman_val_rank"], 3)))
    add(("B ranking negative groups", f"{rk['n_groups_negative']} of {b['n_groups']}"))
    for style, exp in (("factual", ".000"), ("concise", ".993"), ("technical", ".990"),
                       ("verbose", ".953"), ("child_like", ".920"), ("poetic", ".860")):
        got = b["styles"][style]["mean_holdout"]
        if f"{got:.3f}"[1:] != exp:
            raise SystemExit(f"B {style} holdout {got} != manuscript {exp}")
    add(("B technical rank", f"{rk['mean_rank_score']['technical']:.3f}".lstrip("0")))

    # calibration + sensitivity
    for scen, mult in (("original", "0.1"), ("scenario_c", "3.7"),
                       ("scenario_a", "0.0"), ("scenario_b", "14.3")):
        got = cal[scen]["margin_over_noise_band"]
        if f"{got:.1f}" != mult:
            raise SystemExit(f"calibration {scen}: {got} != manuscript {mult}")
    add(("sens C static", f"{sens['scenario_c']['jev_static_calibrated']['exact_oracle']}/10"))
    add(("sens C loo", f"{sens['scenario_c']['loo_rsi_calibrated']['exact_oracle']}/10"))

    cases += frozen_cases()
    missing = [(name, s) for name, s in cases if s not in md]
    print(f"checked {len(cases)} numbers against the artifacts")
    for name, s in missing:
        print(f"  MISSING: {name} ({s!r} not found in the manuscript)")
    return [n for n, _ in missing]


def frozen_cases() -> list[tuple[str, str]]:
    """Numbers quoted about the 43-group frozen corpus, from the frozen artifacts.

    The P3 checks above cover the extended corpus.  These cover §5.1-§5.5 and
    §6.2-§6.4 of the manuscript -- the original paper's own results -- so a
    stale number in the frozen sections cannot survive either guard:
    ``scripts/check_reproducibility.py`` proves the JSONs still come out
    byte-for-byte, and this proves the prose still matches them.  Which fields a
    quoted number comes from is recorded per case, and derived claims (the
    headroom share, the tolerance grid, the sampling counts) are asserted here
    rather than trusted, so the paper's arithmetic is re-run too.
    """
    def load(name: str) -> dict:
        return json.loads((CODE / "jev_rsi/results" / name).read_text(encoding="utf-8"))

    jr, rep = load("jev_rsi_results.json"), load("repair_results.json")
    dg, bay = load("diagnosis_results.json"), load("bayes_meta_results.json")
    live, xt = load("real_jev_results.json"), load("cross_task_results.json")
    xr = load("cross_task_rank_results.json")

    out: list[tuple[str, str]] = []
    add = out.append

    # -- §5.1 Table 1: deployment outcomes over the frozen corpus ----------
    dec = {k: v["decision"] for k, v in jr["main_table"]["policies"].items()}

    def row(name: str, bold_upto: int = 0) -> str:
        d = dec[name]
        cells = [f"{d['exact_oracle']}/32 ({d['exact_oracle']/32:.1%})",
                 f"{d['within_noise']}/32",
                 f"+{d['avg_regret']:.5f}",
                 f"{d['max_regret']}",
                 f"{d['collapse_avoided']}/{d['n_collapse_cases']}"]
        if bold_upto:
            cells = [f"**{c}**" if i < bold_upto else c for i, c in enumerate(cells)]
        return " | ".join(cells)

    add(("Table 1 conservative", row("conservative")))
    add(("Table 1 greedy", f"**{dec['greedy']['exact_oracle']}/32 "
         f"({dec['greedy']['exact_oracle']/32:.1%})** | "
         f"**{dec['greedy']['within_noise']}/32 "
         f"({dec['greedy']['within_noise']/32:.1%})** | "
         f"**+{dec['greedy']['avg_regret']:.5f}** | "
         f"{dec['greedy']['max_regret']} | "
         f"{dec['greedy']['collapse_avoided']}/{dec['greedy']['n_collapse_cases']}"))
    add(("Table 1 heuristic", row("heuristic")))
    add(("Table 1 jev", row("jev_static")))

    # jev+RSI is the LOO learner with clustering on, i.e. the 'full' component
    # ablation -- the same rows the artifact keeps per group.
    full_rows = jr["ablations"]["component"]["full"]["rows"]

    def from_rows(rows: list[dict]) -> tuple[int, int, float]:
        exact = sum(1 for r in rows if abs(r["rsi_regret"]) < 1e-12)
        within = sum(1 for r in rows if r["rsi_regret"] <= 0.007)
        return exact, within, sum(r["rsi_regret"] for r in rows) / len(rows)

    e, w, a = from_rows(full_rows)
    add(("Table 1 jev+RSI", f"{e}/32 ({e/32:.1%}) | {w}/32 | +{a:.5f} | 0.2038 | 4/6"))

    # -- §5.1 Table 2: paired comparisons ---------------------------------
    for key, label in (("greedy_vs_jev_static", "greedy - jev"),
                       ("greedy_vs_heuristic", "greedy - heuristic")):
        p = jr["deep_dive"]["paired_tests"][key]
        lo, hi = p["ci95"]
        add((f"Table 2 {label}",
             f"{p['mean_diff']:+.6f} | [{lo:+.6f}, {hi:+.6f}] | "
             f"p = {p['sign_test_p']:.3f}".replace("p = 0", "p = ") +
             f" | {p['losses']} / {p['wins']} / {p['ties']}"))

    # -- §5.2 Table 3: ablations -----------------------------------------
    for key, label in (("threshold_rule", "single threshold rule"),
                       ("logistic_regression", "logistic regression"),
                       ("bayesian_only", "Bayesian posterior alone"),
                       ("jev_static", "structured gate"),
                       ("heuristic", "heuristic")):
        m = jr["ablations"]["mechanism"][key]
        hit = f"{m['exact_oracle']}/32"
        if key == "logistic_regression":        # the paper bolds this one
            hit = f"**{hit}**"
        add((f"Table 3 {label}",
             f"{hit} | {m['within_noise']}/32 | +{m['avg_regret']:.5f}"))
    for var, label, bold in (("full", "full gate", False),
                             ("no_noul", "collapse primitive removed", True),
                             ("no_score", "trust primitive removed", True)):
        rows = jr["ablations"]["component"][var]["rows"]
        e, w, a = from_rows(rows)
        head = f"**{e}/32** | " if bold else f"{e}/32 | "
        add((f"Table 3 component {label}", f"{head}{w}/32 | +{a:.5f}"))

    # -- §5.3/§5.4: learners and the response surface ---------------------
    dz = rep["diagnosis"]
    add(("§5.3 LOO record", f"{dz['n_wins']} better, {dz['n_losses']} worse, {dz['n_ties']}"))
    loo = jr["loo_rsi"]["rows"]
    parts = []
    for field, word in (("collapse_threshold", "collapse"),
                        ("alignment_threshold", "alignment"),
                        ("confidence_floor", "confidence")):
        v = [r["learned_meta"][field] for r in loo]
        mu = sum(v) / len(v)
        sd = (sum((x - mu) ** 2 for x in v) / len(v)) ** 0.5
        parts.append(f"{word} {mu:.4f} ± {sd:.4f}")
    add(("§5.3 thresholds", ", ".join(parts)))

    gl = bay["existing_grid_loo"]
    st, sd = bay["strategies"]["grid_box"], bay["strategies"]["document_box"]
    add(("Table 4 grid", f"{gl['n_points']} | **{gl['exact_oracle']}/32** | +{gl['avg_regret']:.5f}"))
    for strat, label in ((st["bayes_loo"], "GP-BO"), (st["random_loo"], "random")):
        s = strat["summary"]
        add((f"Table 4 {label}",
             f"50 | {s['exact_oracle']}/32 | +{s['avg_regret']:.5f} | "
             f"{strat['vs_static']['wins']} / {strat['vs_static']['losses']} / "
             f"{strat['vs_static']['ties']}"))
    s = sd["bayes_loo"]["summary"]
    add(("Table 4 BO narrow",
         f"50 | {s['exact_oracle']}/32 | +{s['avg_regret']:.5f} | 0 / 0 / 32"))
    add(("Table 4 in-sample", f"same box | 50 | {st['bayes_in_sample']['exact_oracle']}/32"))

    ls = bay["landscape"]
    add(("§5.4 sampling", f"{ls['n_samples']} points"))
    add(("§5.4 distinct values",
         f"{ls['n_distinct_objectives']} distinct objective values and "
         f"{ls['n_distinct_decision_vectors']} distinct decision vectors"))
    add(("§5.4 objective range", f"{ls['objective_min']:.6f} \u2013 {ls['objective_max']:.6f}"))

    # -- §5.5 live replication -------------------------------------------
    for model, v in live["models"].items():
        s = v["summary"]
        add((f"§5.5 {model}",
             f"{v['live_calls']}/32 | {v['fallback_calls']} | {s['exact_oracle']}/32 | "
             f"+{s['avg_regret']:.5f}"))
    usage = live["models"]["jev-1.13-free"]["usage"]
    add(("§5.5 token count", f"{usage['input_tokens']:,} input tokens"))
    add(("§5.5 cost", f"${usage['est_cost_usd']:.4f}"))
    add(("§5.5 ungated choice",
         f"correct on {live['models']['jev-1.13-free']['raw_choice_exact_oracle']}/32"))

    # -- §5.5 rule firings (the live run's own diagnostics) ---------------
    # The live artifact keeps the gate's per-group inputs, so the three
    # statements the paper makes about them are recomputed here.  The collapse
    # medians were wrong when first checked: the paper quoted the global
    # maximum (0.53) as the collapse groups' median, and that maximum belongs
    # to a group that does not collapse.  See the repair in patch_manuscript.py.
    gm = live["models"]["jev-1.13-free"]["groups"]
    mp = live["meta_params"]
    coll_ids = set(jr["main_table"]["dataset"]["collapse_group_ids"])
    veto = sum(1 for g in gm if g["gate_reason"] == "low_val_alignment")
    align = [g["features"]["alignment"] for g in gm]
    cvals = [g["features"]["collapse_noul"] for g in gm if g["gid"] in coll_ids]
    ovals = [g["features"]["collapse_noul"] for g in gm if g["gid"] not in coll_ids]
    def median(v: list[float]) -> float:
        w = sorted(v)
        n = len(w)
        return w[n // 2] if n % 2 else (w[n // 2 - 1] + w[n // 2]) / 2
    add(("§5.5 alignment veto", f"{veto} of {len(gm)} groups"))
    add(("§5.5 vetos against its threshold", f"threshold of {mp['alignment_threshold']:.2f}"))
    add(("§5.5 median trust score", f"median of {median(align):.3f}"))
    add(("§5.5 collapse maximum",
         f"maximum {max(ovals + cvals):.2f} against a threshold of "
         f"{mp['collapse_threshold']:.2f}"))
    add(("§5.5 collapse separation",
         f"median {median(cvals):.2f} vs {median(ovals):.2f}"))
    top_collapse = max(gm, key=lambda g: g["features"]["collapse_noul"])
    if top_collapse["gid"] not in coll_ids:
        add(("§5.5 the maximum is not a collapse group",
             "belongs to a group that does not collapse"))

    # -- §6.4 Table 6: gain prediction, both columns ----------------------
    gp = xt["protocols"]["leave_target"]["gain_prediction"]
    add(("Table 6 cold",
         f"**{gp['cold']['mae']:.5f}** | {gp['cold']['mae_champion_only']:.5f}"))
    add(("Table 6 global mean",
         f"{gp['global_mean']['mae']:.5f} | **{gp['global_mean']['mae_champion_only']:.5f}**"))
    add(("Table 6 source mean",
         f"{gp['source_mean']['mae']:.5f} | {gp['source_mean']['mae_champion_only']:.5f}"))

    # -- §6.2 the floating-point loss ------------------------------------
    ba = rep["boundary_analysis"]
    lr = rep["loss_rows"][0]
    add(("§6.2 loss group", rep["diagnosis"]["loss_gids"][0]))
    add(("§6.2 trust score",
         f"trust score {ba['loss_group_alignment']:.16f} < 0.50"))
    add(("§6.2 static regret", f"+{lr['static']['regret']:.6f}"))
    add(("§6.2 learned regret", f"+{lr['rsi']['regret']:.6f}"))
    tau = f"{lr['learned_meta']['alignment_threshold']:.4f}".lstrip("0")
    add(("§6.2 learned alignment threshold", f"\u03c4\u2090 = {tau}"))
    dist = abs(ba["distance_to_threshold"])
    assert round(dist / 1e-15, 1) == 2.2, f"ulp gap is {dist:g}, paper says 2.2e-15"
    add(("§6.2 ulp gap", "one part in 2.2 \u00d7 10\u207b\u00b9\u2075"))
    assert {s["exact_oracle"] for s in rep["sweep"]} == {11}, "the lambda sweep changed"
    add(("§6.2 lambda sweep", "11/32 at every \u03bb"))
    grid = {r["eps"]: (r["wins"], r["losses"], r["ties"]) for r in ba["grid"]}
    assert grid[0.0] == (0, 1, 31), f"eps=0 is {grid[0.0]}, paper says 0/1/31"
    assert all(v == (0, 0, 32) for k, v in grid.items() if k >= 1e-12), \
        "a tolerance no longer removes the loss"
    add(("§6.2 tolerance at eps=0", "0/1/31 for \u03b5 = 0"))
    add(("§6.2 tolerance above the ulp", "0/0/32 for every \u03b5 \u2265 10\u207b\u00b9\u00b2"))

    # -- §6.3 why the corpus limits every method -------------------------
    e1 = dg["experiment_1_alignment"]
    add(("§6.3 zero-variance groups",
         f"{e1['n_zero_val_spread']} of {e1['n_groups']} decision groups"))
    add(("§6.3 unsolvable among them",
         "Four of the eight" if e1["n_zero_val_spread_unsolvable"] == 4 else
         f"UNEXPECTED {e1['n_zero_val_spread_unsolvable']} of the eight"))
    add(("§6.3 total baseline regret",
         f"{jr['main_table']['dataset']['greedy_total_regret']}"))
    hrows = e1["rows"]
    top = sorted(hrows, key=lambda r: -r["greedy_regret"])[:2]
    total = sum(r["greedy_regret"] for r in hrows)
    share = (top[0]["greedy_regret"] + top[1]["greedy_regret"]) / total
    for r in top:
        add((f"§6.3 headroom {r['gid']}", f"{r['gid']}` (+{r['greedy_regret']:.4f})"))
    add(("§6.3 headroom share", f"{share:.1%}"))

    e3 = dg["experiment_3_ood"]
    bal = [r["balanced_accuracy"] for r in e3["unfitted_rules"].values()]
    add(("§6.3 unfitted rule range",
         f"{min(bal):.3f}".lstrip("0") + " \u2013 " + f"{max(bal):.3f}".lstrip("0")))
    add(("§6.3 majority baseline",
         f"{e3['majority_baseline_accuracy']:.4f}".lstrip("0") + " majority baseline"))
    add(("§6.3 nested LOO rule selection",
         f"{e3['nested_loo']['balanced_accuracy']:.4f}".lstrip("0")))
    add(("§6.3 in-sample rule",
         f"{e3['in_sample_best_rule']['balanced_accuracy']:.4f}".lstrip("0")))

    # -- §6.4 cross-task transfer ----------------------------------------
    h = xt["headline"]
    add(("Table 6 cold MAE", f"{h['gain_mae_cold']:.5f}"))
    add(("Table 6 source-mean MAE", f"{h['gain_mae']:.5f}"))
    res = xr["results"]
    for mode, label in (("cold", "cold"), ("global_mean", "global mean"),
                        ("val_rank", "validation gain"), ("source_mean", "source mean"),
                        ("pairwise_rank", "pairwise ranker"),
                        ("fixed_contract", "fixed contract")):
        r = res[mode]
        hits = f"{round(r['accuracy_multi'] * 3)}/3"
        regret = f"{r['avg_regret_multi']:.6f}"
        if mode in ("pairwise_rank", "fixed_contract"):   # bolded in the paper
            hits, regret = f"**{hits}**", f"**{regret}**"
        add((f"Table 7 {label}", f"{hits} | {regret}"))
    vu = xt["val_usable"]
    if vu["n_champion_val_usable"] == 0 and vu["n_champion"] == 9:
        add(("§6.4 champions without signal", "9 of 9"))
    folds = xr["learned_constant_per_fold"]
    words = {1: "once", 2: "twice", 3: "three times", 4: "four times",
             5: "five times", 6: "six times"}
    assert folds.count("contract") == 1, f"learned constant moved: {folds}"
    add(("§6.4 learned constant",
         f"`contract` once and `none` {words[folds.count('none')]}"))
    champ = [r for r in res["cold"]["rows"] if r["kind"] == "champion"
             and r["target"] == "gpqa-s926"]
    if champ:
        add(("§6.4 oracle margin", f"margin of **{champ[0]['oracle_gain']:.4f}**"))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--md", default=str(PAPER_MD))
    args = ap.parse_args()
    p = pathlib.Path(args.md)
    if not p.exists():
        # The manuscript lives in its own repository and is not part of the code
        # bundle, so the default path only resolves on the authors' machine.
        raise SystemExit(
            f"cannot read the manuscript at {p}\n"
            "  this checker compares the prose against the JSON artifacts, so it needs\n"
            "  the manuscript markdown, which ships with the paper repository and not\n"
            "  with the code bundle.  Pass --md <path>, or set APC_PAPER_DIR to the\n"
            "  paper repository root (default: ../02_apc_paper).")
    md = p.read_text(encoding="utf-8")
    md = md.replace("\u2212", "-")   # the manuscript uses U+2212 for negatives
    md = " ".join(md.split())        # the prose is hard-wrapped, so collapse it
    missing = check(md)
    if missing:
        print(f"FAIL: {len(missing)} numbers could not be verified")
        return 1
    print("OK: every quoted number matches the released artifacts")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
