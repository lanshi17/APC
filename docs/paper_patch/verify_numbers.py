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

    missing = [(name, s) for name, s in cases if s not in md]
    print(f"checked {len(cases)} numbers against the artifacts")
    for name, s in missing:
        print(f"  MISSING: {name} ({s!r} not found in the manuscript)")
    return [n for n, _ in missing]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--md", default=str(PAPER_MD))
    args = ap.parse_args()
    md = pathlib.Path(args.md).read_text(encoding="utf-8")
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
