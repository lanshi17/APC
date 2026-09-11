"""F11 confirmatory 扩展合并器:n=100 结果并入 real_hle.json + 配对统计。

用法: .venv/bin/python scripts/merge_hle_ext.py [--dry-run]
- 读 experiments/apcbench/real_hle_ext.json(923 批次 6 行)
- 每臂统计:holdout、acc/100、fmt/100、95% Wilson CI
- 跨臂配对:公共 doc 集上的逐题 acc 对齐(gepa vs z0 McNemar exact binomial)
- --dry-run 只打印;否则合并(同 (method,seed) 行去重保留新)并写回主表
"""
from __future__ import annotations

import argparse
import json
import math
import pathlib
import sys
from collections import defaultdict
from itertools import combinations

REPO = pathlib.Path(__file__).resolve().parents[1]
MAIN = REPO / "experiments" / "apcbench" / "real_hle.json"
EXT = REPO / "experiments" / "apcbench" / "real_hle_ext.json"
STREAM = lambda arm: pathlib.Path(f"/tmp/hle_{arm}_923.jsonl")  # noqa: E731
ARMS = ["z0", "math-champ", "contract-champ", "financial-champ", "gepa", "espo"]


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def mcnemar_exact(b: int, c: int) -> float:
    """双侧精确二项(b, c = 不一致对计数)。"""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def load_streams() -> dict[str, dict[str, dict]]:
    out = {}
    for arm in ARMS:
        sp = STREAM(arm)
        if not sp.exists():
            out[arm] = {}
            continue
        by_doc: dict[str, dict] = {}
        for line in sp.read_text(encoding="utf-8").splitlines():
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "<call-error>" in str(r.get("output", "")):
                continue
            if r.get("doc"):
                by_doc[r["doc"]] = r
        out[arm] = by_doc
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    if EXT.exists():
        ext = json.loads(EXT.read_text(encoding="utf-8"))
        rows = {r["method"]: r for r in ext["rows"]}
    else:
        rows = {}
        print("(ext json 未生成——批次未跑完,仅 stream 分析)")
    streams = load_streams()

    print("== per-arm (n per stream, err-rows excluded) ==")
    common_docs = None
    for arm in ARMS:
        st = streams.get(arm, {})
        if not st:
            print(f"{arm:16s} MISSING stream")
            continue
        acc = sum(1 for r in st.values() if float(r.get("accuracy", 0)) > 0.5)
        fmt = sum(1 for r in st.values() if float(r.get("format_score", 0)) > 0.5)
        n = len(st)
        lo, hi = wilson(acc, n)
        r = rows.get(arm, {})
        print(f"{arm:16s} n={n:3d} acc={acc:2d} ({acc/n:.3f} CI[{lo:.3f},{hi:.3f}]) "
              f"fmt={fmt}/{n} json_hold={r.get('holdout_score')}")
        common_docs = set(st) if common_docs is None else (common_docs & set(st))

    print(f"\n== paired common-doc set: {len(common_docs or [])} problems ==")
    if common_docs and len(common_docs) >= 20:
        for a, b in combinations(["z0", "gepa", "espo"], 2):
            ba = streams[a]
            bb = streams[b]
            p10 = sum(1 for d in common_docs if float(ba[d].get("accuracy", 0)) > .5 and float(bb[d].get("accuracy", 0)) <= .5)
            p01 = sum(1 for d in common_docs if float(ba[d].get("accuracy", 0)) <= .5 and float(bb[d].get("accuracy", 0)) > .5)
            pv = mcnemar_exact(p10, p01)
            print(f"McNemar acc {a} vs {b}: {p10}/{p01} p={pv:.3f}")

    if args.dry_run:
        print("\n(dry-run, main table untouched)")
        return 0

    doc = json.loads(MAIN.read_text(encoding="utf-8"))
    assert len(rows) == 6, f"扩展行不全({len(rows)}/6),不合表——等批次完成"
    for arm, r in rows.items():
        doc["rows"] = [x for x in doc["rows"]
                       if not (x["method"] == r["method"] and x.get("seed") == r.get("seed"))] + [r]
    doc.setdefault("notes", []).append(
        f"confirmatory n=100 extension (seed 923, reuse champs) merged {len(rows)} rows; "
        f"pre-registered primary verdict remains the n=30 seed-921 batch")
    MAIN.write_text(json.dumps(doc, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\nmerged {len(rows)} rows -> {MAIN.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
