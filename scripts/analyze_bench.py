# -*- coding: utf-8 -*-
"""APCBench 统计分析：均值±std、配对 bootstrap 95% CI、预算曲线、迁移矩阵。

配对方式：同一 (model_id, seed) 下两方法 holdout 分数之差；
bootstrap 2000 次重采样（numpy RandomState(0)，确定可复现）。
输出 experiments/apcbench/summary.json + 控制台表格。
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
EXP = REPO / "experiments" / "apcbench"
RNG = np.random.RandomState(0)
N_BOOT = 2000


def load(budget: int) -> list[dict]:
    return json.loads((EXP / f"bench_results_b{budget}.json").read_text(encoding="utf-8"))


def paired_diff(rows: list[dict], a: str, b: str) -> tuple[float, tuple[float, float]]:
    """同一 (model, seed) 配对差 a−b 的均值与 bootstrap 95% CI。"""
    key = lambda r: (r["model_id"], r["seed"])
    da = {key(r): r["holdout_score"] for r in rows if r["method"] == a}
    db = {key(r): r["holdout_score"] for r in rows if r["method"] == b}
    common = sorted(set(da) & set(db))
    diffs = np.array([da[k] - db[k] for k in common])
    boots = np.array([np.mean(RNG.choice(diffs, size=len(diffs), replace=True))
                      for _ in range(N_BOOT)])
    return float(np.mean(diffs)), (float(np.percentile(boots, 2.5)),
                                   float(np.percentile(boots, 97.5)))


def method_stats(rows: list[dict]) -> dict:
    out = {}
    gm = defaultdict(list)
    for r in rows:
        gm[r["method"]].append(r["holdout_score"])
    for m, v in sorted(gm.items()):
        a = np.array(v)
        out[m] = {"n": len(v), "mean": round(float(a.mean()), 4),
                  "std": round(float(a.std(ddof=1)) if len(a) > 1 else 0.0, 4),
                  "min": round(float(a.min()), 4), "max": round(float(a.max()), 4)}
    return out


def main():
    summary: dict = {"budgets": {}, "contrasts_b100": {}, "transfer": {},
                 "contract_contrasts": {}, "math_contrasts": {}}
    for b in (25, 50, 100):
        rows = load(b)
        summary["budgets"][str(b)] = method_stats(rows)
    rrows = json.loads((EXP / "robustness_results.json").read_text(encoding="utf-8"))
    rgm = defaultdict(list)
    for r in rrows:
        rgm[r["method"]].append(r["drop"])
    summary["robustness"] = {
        m: {"n": len(v), "mean_drop": round(float(np.mean(v)), 4),
            "std": round(float(np.std(v, ddof=1)), 4)}
        for m, v in sorted(rgm.items())}
    crows = json.loads((EXP / "contract_results.json").read_text(encoding="utf-8"))
    cgm = defaultdict(list)
    for r in crows:
        cgm[r["method"]].append(r["holdout_score"])
    summary["contract"] = {
        m: {"n": len(v), "mean": round(float(np.mean(v)), 4),
            "std": round(float(np.std(v, ddof=1)), 4)}
        for m, v in sorted(cgm.items())}
    for a, b in [("apc-full", "zero-shot"), ("apc-pgam", "zero-shot"),
                 ("apc-full", "manual"), ("apc-pgam", "apc-full")]:
        mean, (lo, hi) = paired_diff(crows, a, b)
        summary["contract_contrasts"][f"{a} - {b}"] = {
            "mean": round(mean, 4), "ci95": [round(lo, 4), round(hi, 4)],
            "significant": bool(lo > 0 or hi < 0)}
    mrows = json.loads((EXP / "math_results.json").read_text(encoding="utf-8"))
    mgm = defaultdict(list)
    for r in mrows:
        mgm[r["method"]].append(r["holdout_score"])
    summary["math"] = {
        m: {"n": len(v), "mean": round(float(np.mean(v)), 4),
            "std": round(float(np.std(v, ddof=1)), 4)}
        for m, v in sorted(mgm.items())}
    summary["math_contrasts"] = {}
    for a, b in [("apc-full", "zero-shot"), ("apc-pgam", "zero-shot"),
                 ("apc-full", "manual"), ("apc-pgam", "apc-full")]:
        mean, (lo, hi) = paired_diff(mrows, a, b)
        summary["math_contrasts"][f"{a} - {b}"] = {
            "mean": round(mean, 4), "ci95": [round(lo, 4), round(hi, 4)],
            "significant": bool(lo > 0 or hi < 0)}
    rows100 = load(100)
    for a, b in [("apc-full", "zero-shot"), ("apc-pgam", "zero-shot"),
                 ("random-search", "zero-shot"), ("apc-full", "manual"),
                 ("apc-pgam", "apc-full"), ("apc-full", "random-search"),
                 ("apc-full", "apc-no-profile"), ("apc-full", "apc-no-halving")]:
        mean, (lo, hi) = paired_diff(rows100, a, b)
        summary["contrasts_b100"][f"{a} - {b}"] = {
            "mean": round(mean, 4), "ci95": [round(lo, 4), round(hi, 4)],
            "significant": bool(lo > 0 or hi < 0)}
    # 跨任务 pooled PGAM−uniform（finance b100 + contract + math，配对差合并）
    pooled = []
    for rows_x in (rows100, crows, mrows):
        ka = {(r["model_id"], r["seed"]): r["holdout_score"] for r in rows_x if r["method"] == "apc-pgam"}
        kb = {(r["model_id"], r["seed"]): r["holdout_score"] for r in rows_x if r["method"] == "apc-full"}
        pooled += [ka[k] - kb[k] for k in sorted(set(ka) & set(kb))]
    pooled = np.array(pooled)
    pboots = np.array([np.mean(RNG.choice(pooled, size=len(pooled), replace=True)) for _ in range(N_BOOT)])
    summary["pgam_pooled"] = {"n": len(pooled), "mean": round(float(pooled.mean()), 4),
                              "ci95": [round(float(np.percentile(pboots, 2.5)), 4),
                                       round(float(np.percentile(pboots, 97.5)), 4)]}
    trows = json.loads((EXP / "transfer_results.json").read_text(encoding="utf-8"))
    for metric in ("decay_direct", "recover"):
        vals = [r[metric] for r in trows if r.get(metric) is not None]
        a = np.array(vals)
        boots = np.array([np.mean(RNG.choice(a, size=len(a), replace=True)) for _ in range(N_BOOT)])
        summary["transfer"][metric] = {
            "n": len(vals), "mean": round(float(a.mean()), 4),
            "ci95": [round(float(np.percentile(boots, 2.5)), 4),
                     round(float(np.percentile(boots, 97.5)), 4)]}
    summary["transfer"]["kr6_pass_rate"] = (
        round(sum(1 for r in trows if r["kr6"]) / len(trows), 4) if trows else None)
    summary["transfer"]["adapt_minus_direct"] = (
        round(float(np.mean([r["adapted"] - r["direct"] for r in trows])), 4))
    (EXP / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1),
                                      encoding="utf-8")
    print("=== method holdout stats ===")
    for b, ms in summary["budgets"].items():
        print(f"-- budget {b} --")
        for m, s in sorted(ms.items()):
            print(f"  {m:15s} n={s['n']:3d} mean={s['mean']:.4f} std={s['std']:.4f} [{s['min']:.4f},{s['max']:.4f}]")
    print("=== paired contrasts @b100 (holdout diff + bootstrap 95% CI) ===")
    for k, c in summary["contrasts_b100"].items():
        sig = "SIGNIFICANT" if c["significant"] else "n.s."
        print(f"  {k:32s} {c['mean']:+.4f} [{c['ci95'][0]:+.4f},{c['ci95'][1]:+.4f}] {sig}")
    print("=== transfer ===")
    for k, c in summary["transfer"].items():
        print(f"  {k}: {c}")


if __name__ == "__main__":
    sys.exit(main())
