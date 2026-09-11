"""HLE-exact 子集生成器(F11 数据可复现)。

上游:Humanity's Last Exam (cais/hle, **MIT**(HF datasets API tags 实测,
license:mit), Nguyen et al. 2025, arXiv 2501.14249)。官方 repo 为 gated(auto);
本脚本经 ungated 文本镜像 macabdul9/hle_text_only 的 datasets-server /rows API
拉取(只读公开端点,分页 100/页),筛 answer_type=='exactMatch' 且 len(answer)≤40。

分层抽样:Mathematics/Physics/Chemistry/Other 各取 30(seed=2026),打乱后
切 dev/validation/holdout = 30/30/30,写 datasets/hle_exact/*.jsonl。

扩展模式 `--extend-holdout N`(F11 power upgrade):保留既有三切分逐行不变,从
各组**剩余池**按 pool 科目比例抽 N 题(seed=2027、排除已用题)追加到 holdout 末尾。
原 30 题 holdout 数字与扩展后同文件前缀兼容(追加式)。

许可:MIT 允许再分发,派生子集随仓提交;本脚本保证 bit 级可再生(seed=2026)。
"""
from __future__ import annotations

import collections
import json
import pathlib
import random
import urllib.parse
import urllib.request

MIRROR = "macabdul9/hle_text_only"
REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = REPO_ROOT / "datasets" / "hle_exact"


def fetch_rows() -> list[dict]:
    repo = urllib.parse.quote(MIRROR, safe="")
    rows: list[dict] = []
    offset = 0
    while offset < 3000:
        u = (f"https://datasets-server.huggingface.co/rows?dataset={repo}"
             f"&config=default&split=test&offset={offset}&length=100")
        try:
            d = json.loads(urllib.request.urlopen(u, timeout=60).read())
        except Exception as e:  # noqa: BLE001
            print("page stop", offset, str(e)[:80])
            break
        batch = [r["row"] for r in d["rows"]]
        if not batch:
            break
        rows += batch
        offset += len(batch)
    return rows


def _write_part(name: str, part: list[dict], start_idx: int = 0) -> None:
    with open(OUT / f"{name}.jsonl", "a" if start_idx else "w", encoding="utf-8") as f:
        for i, r in enumerate(part, start=start_idx):
            f.write(json.dumps({"sample_id": f"hle-{name}-{i}",
                                "input": str(r["question"]),
                                "expected": {"answer": str(r["answer"]).strip()},
                                "subject": r["raw_subject"]}, ensure_ascii=False) + "\n")


def _group_pool(short: list[dict]) -> dict[str, list]:
    groups: dict[str, list] = collections.defaultdict(list)
    for r in short:
        key = r["raw_subject"] if r["raw_subject"] in ("Mathematics", "Physics", "Chemistry") else "Other"
        groups[key].append(r)
    return groups


def extend_holdout(n_extra: int) -> None:
    """追加模式:既有 90 题逐行不动;从剩余池按 pool 科目比例抽 n_extra 题(seed 2027)。"""
    import hashlib
    used: set[str] = set()
    existing = []
    for part in ("dev", "validation", "holdout"):
        for line in (OUT / f"{part}.jsonl").read_text(encoding="utf-8").splitlines():
            rec = json.loads(line)
            used.add(hashlib.sha256(rec["input"].encode()).hexdigest())
            if part == "holdout":
                existing.append(len(line))
    hold_path = OUT / "holdout.jsonl"
    before = hold_path.read_text(encoding="utf-8")
    assert before.endswith("\n") and len(before.splitlines()) == 30, "holdout 非 30 行,先查状态"
    short = [r for r in fetch_rows() if r.get("answer_type") == "exactMatch" and r.get("answer")
             and len(str(r["answer"])) <= 40]
    groups = _group_pool(short)
    pool_n = {k: len(v) for k, v in groups.items()}
    tot = sum(pool_n.values())
    quota = {k: round(n_extra * v / tot) for k, v in pool_n.items()}
    quota["Other"] += n_extra - sum(quota.values())  # 余数给最大组
    rng = random.Random(2027)
    picked: list[dict] = []
    for k in ("Mathematics", "Physics", "Chemistry", "Other"):
        rest = [r for r in groups[k] if hashlib.sha256(str(r["question"]).encode()).hexdigest() not in used]
        rng.shuffle(rest)
        need = min(quota[k], len(rest))
        if need < quota[k]:
            print(f"WARN {k} 剩余池不足 {quota[k]}→{need}")
        picked += rest[:need]
    rng.shuffle(picked)
    assert len(picked) == n_extra, f"实抽 {len(picked)} != {n_extra}"
    with open(hold_path, "a", encoding="utf-8") as f:
        for i, r in enumerate(picked, start=30):
            f.write(json.dumps({"sample_id": f"hle-holdout-{i}",
                                "input": str(r["question"]),
                                "expected": {"answer": str(r["answer"]).strip()},
                                "subject": r["raw_subject"]}, ensure_ascii=False) + "\n")
    print(f"extended holdout 30 -> {30 + len(picked)} (quota {quota})")


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--extend-holdout", type=int, default=0)
    args = ap.parse_args()
    if args.extend_holdout:
        extend_holdout(args.extend_holdout)
        return
    rows = fetch_rows()
    short = [r for r in rows if r.get("answer_type") == "exactMatch" and r.get("answer")
             and len(str(r["answer"])) <= 40]
    rng = random.Random(2026)
    groups = _group_pool(short)
    sel: list[dict] = []
    for k in ("Mathematics", "Physics", "Chemistry", "Other"):
        g = groups.get(k, [])
        rng.shuffle(g)
        sel += g[:30]
    rng.shuffle(sel)
    parts = {"dev": sel[:30], "validation": sel[30:60], "holdout": sel[60:90]}
    OUT.mkdir(parents=True, exist_ok=True)
    for name, part in parts.items():
        _write_part(name, part)
        print(name, len(part))
    print(f"pool exactMatch-short={len(short)}; license upstream=MIT(实测)")


if __name__ == "__main__":
    main()
