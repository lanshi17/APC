# -*- coding: utf-8 -*-
"""聚合真实 LLM 基准结果 → 论文表格(markdown)。

输入: experiments/apcbench/real_<task>.json (bench_real_full.py)
      experiments/apcbench/real_transfer_<src>_to_<tgt>.json (bench_real_transfer.py)
输出: stdout markdown(直接贴进 docs/paper-apc.md §3.4)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

EXP = Path(__file__).resolve().parents[1] / "experiments" / "apcbench"
ORDER = ["zero-shot", "manual", "apc-full"]


def load(p: Path):
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def main() -> int:
    tasks = ["contract", "math", "financial"]
    data = {t: load(EXP / f"real_{t}.json") for t in tasks}
    rows = []
    for t in tasks:
        d = data[t]
        if not d or not d.get("rows"):
            continue
        by = {r["method"]: r for r in d["rows"]}
        for m in ORDER:
            r = by.get(m)
            if r:
                rows.append((t, m, r, d["meta"].get("partial"), r["holdout_score"]))
    mv = next((r["model_version"] for t in tasks if data[t] and data[t]["rows"] for r in data[t]["rows"]), "?")
    print(f"### 真实 LLM 主对比（qwen3.8-flash reasoning，rule-judge 同口径，temp=0）\n")
    print("| 任务 | 方法 | holdout | Δ vs zero-shot | 备注 |")
    print("|---|---|---|---|---|")
    for t, m, r, partial, hold in rows:
        z = next((x for x in rows if x[0] == t and x[1] == "zero-shot"), None)
        delta = f"{hold - z[4]:+.4f}" if z and m != "zero-shot" else "—"
        note = "⚠partial" if partial else ""
        if m == "apc-full":
            note = (note + " " + f"root={r['baseline_score']:.4f}, b={r['budget_used']}").strip()
        print(f"| {t} | {m} | {hold:.4f} | {delta} | {note} |")
    print()

    for p in sorted(EXP.glob("real_transfer_*.json")):
        d = load(p)
        by = {r["method"]: r for r in d["rows"]}
        meta = d["meta"]
        print(f"### 跨任务迁移 {meta['source']} → {meta['target']}（预算 {meta['budget']}）\n")
        print("| 臂 | dev | val | holdout | 预算 |")
        print("|---|---|---|---|---|")
        for m in ("cold", "transfer-0", "transfer-ws"):
            r = by.get(m)
            if r:
                print(f"| {m} | {r['baseline_score']:.4f} | {r['validation_score']:.4f} "
                      f"| {r['holdout_score']:.4f} | {r['budget_used']} |")
        print(f"\n迁移收益 transfer-ws − cold = **{d['transfer_gain_vs_cold']:+.4f}**\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
