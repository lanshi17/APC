# -*- coding: utf-8 -*-
"""GEPA-gpt6 冠军复核：judge-gaming 剥离实验。

背景：bench_real_gepa --model gpt6 --task financial 产出 hold=.8914（z0 同 run 基线 .68 带）。
冠军 prompt 的 <style_rules> 段逐条编码了 rule-judge 判分口径（summary 双模板、metrics
收录规则、"违反将直接导致评分失败"）——疑似反思器逆向 judge 制造伪增益。

双臂（同 judge/checker/samples，holdout20）:
  champ-verbatim  —— 冠军原文重评（验证 .8914 可复现）
  champ-stripped  —— 移除 <style_rules>…</style_rules> 后重评（若掉回 z0 带 → 增益全部来自 gaming 段）
"""
from __future__ import annotations

import json
import re
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO / "apc-pipeline"))

from bench_real_gepa import CHECKER, JUDGE, RolloutBudget, document, eval_cases
from apc.core.task_spec import TaskSpec
from apc.models.factory import create_client

TASK_YAML = "configs/tasks/financial_analysis.yaml"


def main() -> int:
    champ = Path("/tmp/gepa_financial_42_champ.txt").read_text(encoding="utf-8")
    stripped = re.sub(r"<style_rules>.*?</style_rules>\s*", "", champ, flags=re.S)
    print(f"champ {len(champ)} chars -> stripped {len(stripped)} chars", flush=True)

    client = create_client("gpt6", prefer_mock=False)
    if hasattr(client, "client"):
        import httpx
        client.client.timeout = httpx.Timeout(420.0)
    spec = TaskSpec.from_yaml(REPO / TASK_YAML)
    hold = [json.loads(l) for l in (REPO / "datasets" / "financial_analysis" / "holdout.jsonl").read_text(encoding="utf-8").splitlines()][:20]

    rows = []
    for name, text in (("champ-verbatim", champ), ("champ-stripped", stripped)):
        t0 = time.time()
        score, cases = eval_cases(client, spec, hold, text, RolloutBudget(limit=20), hard=480.0)
        acc = sum(c["accuracy"] for c in cases) / len(cases) if cases else 0.0
        fmt = sum(c["format_score"] for c in cases) / len(cases) if cases else 0.0
        rows.append({"method": name, "holdout_score": round(score, 4), "accuracy": round(acc, 4),
                     "format": round(fmt, 4), "n": len(cases), "elapsed_s": round(time.time() - t0, 1),
                     "model_id": client.model_id, "chars": len(text)})
        print(f"{name:15s} hold={score:.4f} acc={acc:.4f} fmt={fmt:.4f} ({rows[-1]['elapsed_s']}s)", flush=True)

    out = REPO / "experiments" / "apcbench" / "real_gepa_gpt6_recheck.json"
    out.write_text(json.dumps({"meta": {"protocol": "gepa-gaming-recheck", "model": "gpt6",
                                        "note": "剥离 <style_rules>(judge 口径编码段) 后增益应消失 → free-text 反思优化器逆向 rule-judge 的直接证据"},
                               "rows": rows}, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"-> {out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
