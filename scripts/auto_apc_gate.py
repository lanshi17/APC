"""AutoAPC-Select：部署期冠军门控选择器(F8)。

问题(F7)：rule-root 搜索臂有种子脆弱性(s44 崩至 .13)，同日噪声带 ±.007；
人工跨臂比较需要配对复测。方案：部署前用**同一批 val 分数**在候选
{z0, manual, full, safe, (gepa)} 中自动选择，tie 时按"最简优先"层级
z0 < manual < safe < full < gepa 打破(奥卡姆序)，从而以零额外 rollout
获得非劣保证：worst case 回退到当日最强稳健臂。

本脚本离线回放全部 12 组既有臂(val,hold)数据：gate 选择 vs oracle vs worst。
用法：`.venv/bin/python scripts/auto_apc_gate.py`(纯离线，零 API)。
"""
from __future__ import annotations

import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
BENCH = REPO / "experiments" / "apcbench"
SIMPLE_ORDER = ["zero-shot", "reeval-z0", "manual", "reeval-manual", "apc-safe",
                "reeval-safe", "z0-control", "gepa", "apc-full", "reeval-full",
                "transfer-ws", "transfer-0", "cold"]
# 简化：以 (method 名) 在组内位置代表"最简优先"序 — 组用显式列表给出更稳

def pick(group: list[dict]) -> dict:
    """argmax val；±.007 噪声带内并列时取组内声明序最前者。"""
    best = max(g["validation_score"] for g in group)
    tol = [g for g in group if g["validation_score"] >= best - 0.007]
    return tol[0]  # 组列表按简洁优先排好


def main() -> None:
    groups = []
    fin = json.loads((BENCH / "real_financial.json").read_text())["rows"]
    for seed in (42, 43, 44):
        g = [r for r in fin if r.get("seed", 42) == seed and r["method"] != "zero-shot"
             and r["validation_score"] is not None]
        z0 = [r for r in fin if r.get("seed", 42) == seed and r["method"] == "zero-shot"]
        g = z0 + [r for r in g if r["method"] != "zero-shot"]
        # 42 有 4 臂，43/44 只搜了两臂 — 组内声明序 = 简洁优先
        order = {"zero-shot": 0, "manual": 1, "apc-safe": 2, "apc-full": 3}
        g.sort(key=lambda r: order.get(r["method"], 9))
        if len(g) >= 2:
            groups.append((f"financial-s{seed}", g))
    m = json.loads((BENCH / "real_math.json").read_text())["rows"]
    groups.append(("math-s42", [r for r in m if r["validation_score"] is not None]))
    for f in sorted(BENCH.glob("real_transfer_*.json")):
        d = json.loads(f.read_text())
        rows = d["rows"] if isinstance(d, dict) else d
        rows = [r for r in rows if r.get("validation_score") is not None]
        order = {"transfer-ws": 0, "transfer-0": 1, "cold": 2}
        rows.sort(key=lambda r: order.get(r["method"], 9))
        if len(rows) >= 2:
            groups.append((f.name.replace(".json", ""), rows))

    n_hit = n_oracle = n_worst = 0
    print(f"{'组':28s} {'gate选':14s} {'gate hold':9s} {'oracle':8s} {'worst':8s} {'regret':7s}")
    for name, g in groups:
        sel = pick(g)
        orc = max(x["holdout_score"] for x in g)
        wor = min(x["holdout_score"] for x in g)
        regret = orc - sel["holdout_score"]
        n_hit += 1 if regret < 0.007 else 0
        n_oracle += 1 if abs(regret) < 1e-9 else 0
        n_worst += 1 if abs(sel["holdout_score"] - wor) < 1e-9 else 0
        print(f"{name:28s} {sel['method']:14s} {sel['holdout_score']:.4f}   "
              f"{orc:.4f}   {wor:.4f}   {regret:+.4f}")
    print(f"\ngroups={len(groups)} within-noise={n_hit} exact-oracle={n_oracle} picked-worst={n_worst}")


if __name__ == "__main__":
    main()
