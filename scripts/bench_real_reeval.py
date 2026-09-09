"""APC 三 genome(z0=base / apc-safe 冠军 / apc-full 冠军)+ GEPA 冠军文本的
**同日配对 holdout 复测** — 消除 reasoning 模型日漂移混淆(variance 审计)。

复用 bench_real_gepa.eval_cases(TrialScorer 同构计分)。结果并入 real_gepa.json,
method ∈ {reeval-z0, reeval-safe, reeval-full, reeval-gepa},seed=900+当日序号。
"""
from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "apc-pipeline"))
sys.path.insert(0, str(REPO / "scripts"))

from apc.core.genome import PromptGenome
from apc.core.task_spec import TaskSpec
from apc.compiler.renderer import DefaultPromptCompiler
from apc.models.factory import create_client
from apc.models.mock_client import MockClient

from bench_real_full import TASK_CFG, BASE_GENOME, CHAMPS_DIR, real_profile, manual_genome  # noqa: E402
from bench_real_gepa import eval_cases, RolloutBudget, OUT  # noqa: E402

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True, choices=list(TASK_CFG))
    ap.add_argument("--model", default="qwen")
    ap.add_argument("--seed", type=int, default=902, help="当日序号(902=第2日)")
    ap.add_argument("--gepa-champ", default=None, help="GEPA 冠军文本文件(可选第4臂)")
    ap.add_argument("--only", default=None, help="逗号分隔臂名(如 reeval-manual),缺省全跑")
    args = ap.parse_args()

    client = create_client(args.model, prefer_mock=False)
    if isinstance(client, MockClient):
        raise SystemExit("需要真实凭证")
    if hasattr(client, "client"):
        import httpx
        client.client.timeout = httpx.Timeout(420.0)
    task_yaml, ds_name, judge_id = TASK_CFG[args.task]
    spec = TaskSpec.from_yaml(REPO / task_yaml)
    compiler = DefaultPromptCompiler()
    from apc.core.model_profile import ModelProfile
    from apc.models.mock_client import MockClient as _MC
    profile = (ModelProfile.from_json(Path(str(REPO / "artifacts/profiles/qwen_probe.json")).read_text(encoding="utf-8"))
               if isinstance(client, _MC) else real_profile(client))
    from apc.evaluation.dataset import load_dataset
    hold = load_dataset(REPO / "datasets" / ds_name / "holdout.jsonl").samples[:20]

    doc = {"protocol": "real-gepa", "judge": judge_id, "rows": []}
    if OUT.exists():
        doc = json.loads(OUT.read_text(encoding="utf-8"))

    def champ(safe: bool) -> PromptGenome:
        fn = f"real_{args.task}_champ_safe.json" if safe else f"real_{args.task}_champ.json"
        return PromptGenome.model_validate_json((CHAMPS_DIR / fn).read_text(encoding="utf-8"))

    base = PromptGenome.from_json(str(REPO / BASE_GENOME))
    arms = [("reeval-z0", base),
            ("reeval-manual", manual_genome(base)),
            ("reeval-safe", champ(True)),
            ("reeval-full", champ(False))]
    if args.gepa_champ:
        arms.append(("reeval-gepa", None))
    for method, g in arms:
        if args.only and method not in {x.strip() for x in args.only.split(",")}:
            continue
        t0 = time.time()
        if g is None:
            text = Path(args.gepa_champ).read_text(encoding="utf-8")
        else:
            text = compiler.compile(g, spec, profile, apply_rules=False).prompt_text
        sc, cases = eval_cases(client, spec, hold, text, RolloutBudget(10_000))
        row = {"task": args.task, "seed": args.seed, "method": method,
               "holdout_score": round(sc, 4), "validation_score": None,
               "pool_size": 1, "iterations": 0, "rollouts_used": len(cases),
               "elapsed_s": round(time.time() - t0, 1), "model_version": client.model_version,
               "prompt_chars": len(text),
               "fail_sample": [c for c in cases if c["accuracy"] < 0.99][:8]}
        doc["rows"] = [r for r in doc["rows"]
                       if not (r["task"] == args.task and r.get("seed") == args.seed and r["method"] == method)] + [row]
        OUT.write_text(json.dumps(doc, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"{method} {args.task} hold={sc:.4f} ({row['elapsed_s']}s)", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
