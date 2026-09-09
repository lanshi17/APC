"""HLE-exact 子集生成器(F11 数据可复现)。

上游:Humanity's Last Exam (cais/hle, **CC-BY-NC-4.0**, Nguyen et al. 2025,
arXiv 2501.14249)。官方 repo 为 gated(auto);本脚本经 ungated 文本镜像
macabdul9/hle_text_only 的 datasets-server /rows API 拉取(只读公开端点,
分页 100/页),筛 answer_type=='exactMatch' 且 len(answer)≤40。

分层抽样:Mathematics/Physics/Chemistry/Other 各取 30(seed=2026),打乱后
切 dev/validation/holdout = 30/30/30,写 datasets/hle_exact/*.jsonl。

注意:因上游 CC-BY-NC,再分发受限 — 本仓库仅随仓**生成器**与派生子集;
对外发布复现包时以脚本现场重建为准。
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


def main() -> None:
    rows = fetch_rows()
    short = [r for r in rows if r.get("answer_type") == "exactMatch" and r.get("answer")
             and len(str(r["answer"])) <= 40]
    rng = random.Random(2026)
    groups: dict[str, list] = collections.defaultdict(list)
    for r in short:
        key = r["raw_subject"] if r["raw_subject"] in ("Mathematics", "Physics", "Chemistry") else "Other"
        groups[key].append(r)
    sel: list[dict] = []
    for k in ("Mathematics", "Physics", "Chemistry", "Other"):
        g = groups.get(k, [])
        rng.shuffle(g)
        sel += g[:30]
    rng.shuffle(sel)
    parts = {"dev": sel[:30], "validation": sel[30:60], "holdout": sel[60:90]}
    OUT.mkdir(parents=True, exist_ok=True)
    for name, part in parts.items():
        with open(OUT / f"{name}.jsonl", "w", encoding="utf-8") as f:
            for i, r in enumerate(part):
                f.write(json.dumps({"sample_id": f"hle-{name}-{i}",
                                    "input": str(r["question"]),
                                    "expected": {"answer": str(r["answer"]).strip()},
                                    "subject": r["raw_subject"]}, ensure_ascii=False) + "\n")
        print(name, len(part))
    print(f"pool exactMatch-short={len(short)}; license upstream=CC-BY-NC-4.0")


if __name__ == "__main__":
    main()
