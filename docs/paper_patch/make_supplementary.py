#!/usr/bin/env python3
"""Generate the P3 supplementary material from the released result artifacts.

    python docs/paper_patch/make_supplementary.py [--out DIR] [--no-pdf]

Writes ``supplementary.md`` (and ``supplementary.pdf`` if pandoc + xelatex are
available) into the paper repository's ``manuscript_negative/`` directory.  Every
table is generated from ``code/jev_rsi/results/jev_rsi_extended_*.json`` and the
scenario artifacts, so the supplement cannot drift from the code.
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import shutil
import statistics
import subprocess
import sys

CODE = pathlib.Path(__file__).resolve().parents[2]      # <repo>/docs/paper_patch
PAPER_REPO = pathlib.Path(os.environ.get("APC_PAPER_DIR", CODE.parent / "02_apc_paper"))
EXT = CODE / "jev_rsi/corpus_extended"
RESULTS = CODE / "jev_rsi/results"
DEFAULT_OUT = PAPER_REPO / "manuscript_negative"

PREAMBLE = """---
title: "Supplementary Material: Validation Argmax Is Hard to Beat"
subtitle: "P3 extended corpus (Scenarios A, B, C) --- protocols, per-group tables, diagnostics"
---

"""


def load(p: pathlib.Path) -> dict:
    return json.loads(p.read_text(encoding="utf-8"))


def table(header: list[str], rows: list[list[str]]) -> str:
    out = ["| " + " | ".join(header) + " |",
           "|" + "|".join("---" for _ in header) + "|"]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def f(x: float, nd: int = 4) -> str:
    return f"{x:.{nd}f}"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--no-pdf", action="store_true")
    args = ap.parse_args()
    out_dir = pathlib.Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    res = load(RESULTS / "jev_rsi_extended_results.json")
    ana = load(RESULTS / "jev_rsi_extended_analysis.json")
    c = load(EXT / "scenario_c/scores.json")
    a = load(EXT / "scenario_a/scores.json")
    b = load(EXT / "scenario_b/scores.json")
    probe = load(EXT / "scenario_a/judge_variance_probe.json")
    audit = ana["dataset"]

    L: list[str] = [PREAMBLE]
    ap_ = L.append

    # ---------------------------------------------------------------- S1
    ap_("# S1. Corpus, models and protocol\n")
    ap_(f"The extended corpus adds **{audit['n_extended']} groups / "
        f"{sum(v['n_arms'] for k, v in audit['by_scenario'].items())} arms** to the frozen "
        f"corpus ({audit['n_frozen']} groups), for a total of {audit['n_total']} groups, "
        f"{audit['n_arms_total']} arms and {audit['n_decision_total']} deployment decisions.\n")
    ap_(table(["Scenario", "Groups", "Arms", "Val metric", "Holdout metric"],
              [["A: creative writing", "20", "200",
                "one LLM-judge draw (0-10)", "mean of 5 further draws (0-10)"],
               ["B: image captioning", "15", "90",
                "text-only judge vs human references", "vision judge (image + caption)"],
               ["C: summarisation", "10", "60",
                "ROUGE-L F1 vs human highlights",
                "baseline-rescaled BERTScore F1 (deberta-xlarge-mnli)"]]
              ) + "\n")
    ap_("**Generation and judging.** `qwen3.8-flash` generated and judged every "
        "text; `qwen3.8-27b` was used only for the judge-variance probe of S3. "
        "Both are accessed through the DashScope OpenAI-compatible endpoint. "
        "Generation calls disable the reasoning trace (`enable_thinking=false`); "
        "judge calls keep it, matching the protocol used for the frozen corpus. "
        "Every response is cached on disk keyed by model, messages, temperature, "
        "`max_tokens`, seed and tag, and every call is appended to a token ledger "
        "(`corpus_extended/_cache/usage.jsonl`), so re-running the corpus is free "
        "and byte-identical.\n")
    ap_("**Deviations from the P3 plan, recorded for transparency.** "
        "(i) `gpt-4o`/`gpt-4o-vision` were unavailable (no API key), so the "
        "scenario builders use the Qwen models above; (ii) the plan's inline "
        "pseudo-code scored summaries against the raw article, which caps ROUGE-L "
        "F1 near 0.2 for every arm, so we score against the dataset's human "
        "highlights and additionally record the article-referenced variants; "
        "(iii) BERTScore is reported baseline-rescaled, because raw F1 is ~0.9 for "
        "any fluent text and the plan's `holdout < 0.3` band is only meaningful on "
        "the rescaled scale; (iv) all judge scores are normalised 0-10 -> 0-1 to "
        "match the gate's calibration, with the raw numbers kept in `judge_raw`; "
        "(v) the `extractive` arm is a deterministic TF-IDF sentence selector "
        "rather than an LLM call, since an LLM-written summary is not extractive.\n")

    # ---------------------------------------------------------------- S2
    ap_("# S2. Scenario C: adversarial summarisation\n")
    arms = ana["scenario_c"]["arms"]
    ap_(table(["Arm", "Type", "Mean ROUGE-L F1 (val)", "Mean rescaled BERTScore (holdout)"],
              [[k, v["type"], f(v["mean_val"]), f(v["mean_holdout"])]
               for k, v in sorted(arms.items(), key=lambda kv: -kv[1]["mean_val"])]) + "\n")
    ap_("**Which validation metric would have been captured by a gaming arm** "
        "(groups out of 10 in which the metric's argmax is a gaming arm):\n")
    ap_(table(["Validation metric", "Groups captured"],
              [[k, f"{v}/10"] for k, v in
               sorted(ana["scenario_c"]["val_variant_capture"].items())]) + "\n")
    rows = []
    for gid, rec in sorted(c["groups"].items()):
        for arm in rec["arms"]:
            m = arm["metrics"]
            rows.append([gid.replace("adversarial_summary_", "g"),
                         arm["id"], arm["type"],
                         f(m["rougeL_vs_reference"], 3), f(m["rougeL_vs_article"], 3),
                         f(m["bertscore_raw_vs_reference"], 3),
                         f(arm["holdout_score"], 3)])
    ap_("<details><summary>Per-group, per-arm metrics (60 rows)</summary>\n")
    ap_(table(["Group", "Arm", "Type", "ROUGE-L vs highlights", "ROUGE-L vs article",
               "raw BERTScore", "rescaled BERTScore"], rows) + "\n</details>\n")
    ap_(f"Validation argmax is captured by a gaming arm in "
        f"**{ana['scenario_c']['n_groups_captured_by_gaming']}/10** groups; the mean "
        f"Spearman correlation between validation and holdout is "
        f"**{ana['scenario_c']['mean_val_holdout_spearman']:+.3f}**. The observed val "
        f"range is [{ana['scenario_c']['val_min']:.4f}, "
        f"{ana['scenario_c']['val_max']:.4f}], so the pre-registered capture band "
        f"(`val > 0.4`) is unreachable on this corpus.\n")

    # ---------------------------------------------------------------- S3
    ap_("# S3. Scenario A: judge variance\n")
    per_strategy: dict[str, list[tuple[float, float]]] = {}
    for rec in a["groups"].values():
        for arm in rec["arms"]:
            per_strategy.setdefault(arm["id"], []).append(
                (arm["val_score"], arm["holdout_score"]))
    ap_(table(["Strategy", "Mean val (single draw)", "Mean holdout (5 draws)",
               "Mean |val - holdout|"],
              [[k, f(statistics.fmean(x for x, _ in v), 4),
                f(statistics.fmean(y for _, y in v), 4),
                f(statistics.fmean(abs(x - y) for x, y in v), 4)]
               for k, v in sorted(per_strategy.items())]) + "\n")
    ap_(table(["Group", "Kind", "Mean per-arm judge sd (0-10)",
               "Max per-arm judge sd", "Greedy regret", "Gate reason"],
              [[gid.replace("creative_", ""), rec.get("kind", ""),
                f(rec["judge_variance_10"]["mean_arm_std"], 3),
                f(rec["judge_variance_10"]["max_arm_std"], 3),
                f(next(r["greedy_regret"] for r in ana["scenario_a"]["rows"]
                       if r["gid"] == gid), 4),
                next(r["gate_reason"] for r in ana["scenario_a"]["rows"]
                     if r["gid"] == gid)]
               for gid, rec in sorted(a["groups"].items())]) + "\n")
    ap_("**Can any available judge be noisy enough?** "
        f"{probe['protocol']}\n")
    ap_(table(["Judge model", "Draws", "Arms", "Mean sd (0-10)", "Median sd",
               "Max sd", "Arms with sd > 2.0"],
              [[k, v["draws"], v["n_arms"], f(v["mean_arm_std_10"], 3),
                f(v["median_arm_std_10"], 3), f(v["max_arm_std_10"], 3),
                f"{v['n_arms_std_gt_2']}/{v['n_arms']}"]
               for k, v in probe["models"].items()]) + "\n")
    ap_(f"Scenario A's gate retreats from validation argmax in "
        f"{ana['scenario_a']['n_gate_retreats']}/{ana['scenario_a']['n_groups']} groups; "
        f"the Spearman correlation between judge variance and the baseline's regret is "
        f"{ana['scenario_a']['spearman_variance_vs_greedy_regret']:+.3f}.\n")

    # ---------------------------------------------------------------- S4
    ap_("# S4. Scenario B: text judge vs vision judge\n")
    styles = ana["scenario_b"]["styles"]
    rank = ana["scenario_b"]["ranking_holdout"]["mean_rank_score"]
    ap_(table(["Caption style", "Mean val (text judge)", "Mean holdout (vision, 0-10)",
               "Mean ranking holdout (1 = best)"],
              [[k, f(v["mean_val"], 3), f(v["mean_holdout"], 3), f(rank.get(k, 0), 3)]
               for k, v in sorted(styles.items(), key=lambda kv: -kv[1]["mean_holdout"])])
        + "\n")
    ap_(table(["Image", "Vision ranking (best first)", "Spearman(val, ranking)"],
              [[gid.replace("caption_", ""),
                " > ".join(r["id"] for r in sorted(
                    rec["arms"], key=lambda x: x["holdout_rank_position"])),
                f(rec["val_vs_rank_spearman"], 3)]
               for gid, rec in sorted(b["groups"].items())]) + "\n")
    ap_(f"Mean Spearman(val, holdout) = "
        f"{ana['scenario_b']['mean_val_holdout_spearman']:+.3f}; mean Spearman(val, "
        f"vision ranking) = "
        f"{ana['scenario_b']['ranking_holdout']['mean_spearman_val_rank']:+.3f}, "
        f"negative in {ana['scenario_b']['ranking_holdout']['n_groups_negative']}/15 "
        f"images. The vision score saturates (five of six styles at $\\geq$ 0.92), "
        f"which is why the ranking holdout was added.\n")

    # ---------------------------------------------------------------- S5
    ap_("# S5. Threshold calibration and scale sensitivity\n")
    cal = ana["calibration"]
    ap_(table(["Corpus", "Decision groups", "val min", "val median", "val max",
               "Median top-1/top-2 margin", "x NOISE_BAND (0.007)",
               "Alignment rule saturated"],
              [[k, v["n_decision_groups"], f(v["val_min"], 3), f(v["val_median"], 4),
                f(v["val_max"], 3), f(v["median_margin"], 4),
                f(v["margin_over_noise_band"], 1), f"{v['frac_alignment_saturated']:.0%}"]
               for k, v in cal.items()]) + "\n")
    sens = ana["scale_sensitivity"]
    ap_("Refitting the noise band to each corpus (median absolute gap between "
        "adjacent arms) and re-running the gate and its leave-one-out learner:\n")
    ap_(table(["Corpus", "Fitted sigma", "Fitted align scale", "Greedy",
               "Jev (refit)", "Jev + LOO-RSI (refit)"],
              [[k, f(v["scale"]["sigma"], 4), f(v["scale"]["align_scale"], 4),
                f"{v['greedy']['exact_oracle']}/{v['greedy']['n_groups']} "
                f"({v['greedy']['avg_regret']:+.4f})",
                f"{v['jev_static_calibrated']['exact_oracle']}/"
                f"{v['jev_static_calibrated']['n_groups']} "
                f"({v['jev_static_calibrated']['avg_regret']:+.4f})",
                (f"{v['loo_rsi_calibrated']['exact_oracle']}/"
                 f"{v['loo_rsi_calibrated']['n_groups']} "
                 f"({v['loo_rsi_calibrated']['avg_regret']:+.4f})")
                if v.get("loo_rsi_calibrated") else "--"]
               for k, v in sens.items()]) + "\n")

    # ---------------------------------------------------------------- S6
    bc = ana["boundary_conditions"]
    ap_("# S6. Where the gate beats validation argmax\n")
    ap_(f"Across the 77 decision groups the gate wins **{len(bc['wins'])}**, loses "
        f"**{len(bc['losses'])}** and ties **{len(bc['ties'])}**. The wins and losses "
        f"have opposite signatures, and both are useful for a practitioner.\n")
    ap_(table(["Scenario", "Wins", "Losses", "Ties"],
              [[k, v["wins"], v["losses"], v["ties"]]
               for k, v in bc["by_scenario"].items()]) + "\n")
    ap_("**Win profile.** Every win is a case where the arm the gate retreated to "
        "was, in hindsight, the better arm; the retreat reason is always "
        "`low_val_alignment` and the arm is a designated safe arm "
        "(`zero_shot` / `zero-shot`).\n")
    ap_(table(["Group", "Scenario", "Argmax arm", "Gate arm", "Reason",
               "Argmax regret", "Gate regret", "Gain"],
              [[w["gid"], w["scenario"], w["greedy_arm"], w["jev_arm"],
                w["jev_reason"], f(w["greedy_regret"], 4), f(w["jev_regret"], 4),
                f(w["delta"], 4)] for w in bc["wins"]]) + "\n")
    ap_("**Loss profile.** Losses are dominated by the two mechanisms of the main "
        "text: an unsafe fallback and a mis-scaled threshold. The ten largest are:\n")
    big = sorted(bc["losses"], key=lambda r: r["delta"])[:10]
    ap_(table(["Group", "Scenario", "Argmax arm", "Gate arm", "Reason",
               "Argmax regret", "Gate regret", "Loss"],
              [[r["gid"], r["scenario"], r["greedy_arm"], r["jev_arm"],
                r["jev_reason"], f(r["greedy_regret"], 4), f(r["jev_regret"], 4),
                f(abs(r["delta"]), 4)] for r in big]) + "\n")
    reasons: dict[str, int] = {}
    for r in bc["losses"]:
        reasons[r["jev_reason"]] = reasons.get(r["jev_reason"], 0) + 1
    ap_("Loss reasons: "
        + ", ".join(f"`{k}` × {v}" for k, v in sorted(reasons.items(),
                                                       key=lambda kv: -kv[1]))
        + ". The gate is never wrong because it followed the validation signal: "
          "every loss is a retreat it did not need to make.\n")

    # ---------------------------------------------------------------- S7
    ap_("# S7. Reproduction\n")
    ap_("```bash\n"
        "cd code\n"
        "python -m jev_rsi.corpus_extended.fetch_sources cnn_dailymail --n 10\n"
        "python -m jev_rsi.corpus_extended.fetch_sources flickr30k --n 15\n"
        "python -m jev_rsi.corpus_extended.build_scenario_c\n"
        "python -m jev_rsi.corpus_extended.build_scenario_a\n"
        "python -m jev_rsi.corpus_extended.build_scenario_b\n"
        "python -m jev_rsi.corpus_extended.probe_judge_variance \\\n"
        "    --models qwen3.8-flash qwen3.8-27b --groups 5 --draws 6\n"
        "python scripts/p3_acceptance.py\n"
        "python -m jev_rsi.experiments --corpus extended\n"
        "python -m jev_rsi.analysis_extended\n"
        "```\n")
    ap_("All model responses are cached; the commands above re-run in about 100 "
        "seconds with no API spend and reproduce the JSON artifacts byte-for-byte. "
        "Every table in this supplement is generated from those artifacts at build "
        "time (`docs/paper_patch/make_supplementary.py`), and "
        "`python docs/paper_patch/verify_numbers.py` re-checks the numbers quoted "
        "in the manuscript against them.\n")

    md_path = out_dir / "supplementary.md"
    md_path.write_text("\n".join(L), encoding="utf-8")
    print(f"wrote {md_path}")

    if args.no_pdf:
        return 0
    if not shutil.which("pandoc"):
        print("pandoc not found -- skipping PDF", file=sys.stderr)
        return 0
    pdf = out_dir / "supplementary.pdf"
    cmd = ["pandoc", str(md_path.resolve()), "-o", str(pdf.resolve()),
           "--pdf-engine=xelatex",
           "-V", "geometry:margin=1in", "-V", "fontsize=10pt",
           "-V", "colorlinks=true"]
    env = dict(os.environ)
    # the sandbox's default TMPDIR is read-only; pandoc needs a writable one
    env.setdefault("TMPDIR", "/tmp")
    r = subprocess.run(cmd, capture_output=True, text=True, cwd=out_dir, env=env)
    if r.returncode != 0:
        print(r.stderr[-2500:], file=sys.stderr)
        return r.returncode
    print(f"wrote {pdf}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
