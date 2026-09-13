#!/usr/bin/env python3
"""GPQA-Diamond 六臂合并:stream 权威逐行统计 + 五臂 JSON 汇总。

- Wilson 95% CI(逐臂,acc-only 于名义 190)
- 同文本地板:gepa-926(champ=z0_text 重建,931 字符)vs z0-927 —— 同一 prompt 文本
  两次独立评测的逐题不一致率 = GPQA 上的 temp=0 噪声地板
- McNemar:gepa vs z0(公共 doc 集,双侧二项)
- 完整性:每 stream 190 行名义;缺臂/缺 stream 优雅降级
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
TAG, SEED = "gpqa", 926
ARMS = ["z0", "math-champ", "contract-champ", "financial-champ", "gepa", "espo"]
STREAMS = {"gepa": f"/tmp/{TAG}_gepa_{SEED}.jsonl", "z0-927": f"/tmp/{TAG}_z0_927.jsonl"}
JSONS = {a: REPO / "experiments" / "apcbench" / f"gpqa_{a}.json" for a in ARMS}


def wilson(k: int, n: int) -> tuple[float, float]:
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    z = 1.959963984540054
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def mcnemar(b: int, c: int) -> float:
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    return min(1.0, sum(math.comb(n, i) for i in range(0, k + 1)) / 2 ** n * 2)


def load_stream(path: str) -> dict[str, dict]:
    rows = [json.loads(l) for l in Path(path).read_text(encoding="utf-8").split("\n") if l.strip()]
    return {r["doc"]: r for r in rows}


def main() -> int:
    dry = "--dry-run" in sys.argv
    print("== per-arm (from JSONs, n=190 nominal) ==")
    summary = {}
    for a in ARMS:
        if not JSONS[a].exists():
            print(f"{a:16s} (json pending — arm still running)")
            continue
        r = json.loads(JSONS[a].read_text(encoding="utf-8"))["rows"][-1]
        lo, hi = wilson(r["n_acc"], r["n"])
        summary[a] = r
        print(f"{a:16s} acc={r['n_acc']:3d}/{r['n']} ({r['n_acc']/r['n']:.3f} CI[{lo:.3f},{hi:.3f}]) "
              f"hold={r['holdout_score']:.4f} chars={r['prompt_chars']}")
    if len(summary) >= 2:
        spread = max(s["holdout_score"] for s in summary.values()) - min(s["holdout_score"] for s in summary.values())
        print(f"holdout_score spread: {spread:.4f}")

    streams = {}
    for nm, path in STREAMS.items():
        try:
            streams[nm] = load_stream(path)
            print(f"stream {nm}: {len(streams[nm])} rows")
        except FileNotFoundError:
            print(f"stream {nm}: MISSING (skip paired stats)")

    if "gepa" in streams and "z0-927" in streams:
        g, z = streams["gepa"], streams["z0-927"]
        common = set(g) & set(z)
        print(f"\n== paired common docs: {len(common)} ==")
        g_only = sum(1 for d in common if g[d]["accuracy"] and not z[d]["accuracy"])
        z_only = sum(1 for d in common if z[d]["accuracy"] and not g[d]["accuracy"])
        both = sum(1 for d in common if g[d]["accuracy"] and z[d]["accuracy"])
        p = mcnemar(g_only, z_only)
        print(f"McNemar gepa vs z0: +{g_only}/-{z_only} p={p:.3f} both-right={both}")
        print(f"same-text floor (gepa-926 == z0_text; z0-927 same text): "
              f"{g_only + z_only}/{len(common)} = {(g_only + z_only) / len(common):.3f}")

    if dry:
        print("\n(dry-run, nothing merged)")
        return 0

    out = REPO / "experiments" / "apcbench" / "real_gpqa.json"
    doc = {
        "protocol": "real-gpqa-mcq",
        "judge": "exact-match-normalized-v3.2",
        "dataset": "GPQA-Diamond (fingertap mirror, 198 rows; dev8/holdout190, seed-2027 split, zero overlap)",
        "spec": "external_mcq_v1",
        "notes": [
            "gepa-926 champion reconstructed as z0_text (931 chars) — search accepted zero candidates in 28 iters "
            "(parent_chars constant in search log); stream loss in host reboot, seed identity verified by length",
            "pre-registered verdicts remain HLE n=30 (seed 921); GPQA round is a second-dataset replication",
        ],
        "rows": [json.loads(JSONS[a].read_text(encoding="utf-8"))["rows"][-1]
                 for a in ARMS if JSONS[a].exists()],
    }
    out.write_text(json.dumps(doc, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\nmerged {len(doc['rows'])} rows -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
