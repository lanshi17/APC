"""F11-C · genome 位点归因消融(零 API 构造 + 小样本真实评测)。

问题:APC 的可审计性主张(C1)要求知道 holdout 分数各来自哪个基因位点——
financial 饱和与 HLE 知识墙下 base 分数高,但"编译产物哪些部分在扛分"从未
逐位点量化。本消融把 base genome 逐位点关到惰性值,在 HLE holdout 前 8 题
(与主表同一确定性顺序、同 hardened v3.2 判分)重测。

用法(真实评测需凭证; --emit 只构造文本零成本):
  .venv/bin/python scripts/bench_hle_ablation.py --emit
  .venv/bin/python scripts/bench_hle_ablation.py --loci role,output,verification --seed 924
  .venv/bin/python scripts/bench_hle_ablation.py --loci reasoning,layout,constraints --seed 924
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import pathlib
import sys
import time

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "apc-pipeline"))
sys.path.insert(0, str(REPO / "scripts"))

from apc.core.genome import PromptGenome
from apc.core.task_spec import TaskSpec
from apc.core.model_profile import ModelProfile
from apc.compiler.renderer import DefaultPromptCompiler

import bench_real_hle as H  # noqa: E402  (复用 load/compile_for/run_eval/monkeypatch)
from bench_real_gepa import RolloutBudget  # noqa: E402

# 位点消融表:关掉 base 中实际生效的构件(惰性值)
ABLATIONS: dict[str, dict] = {
    "role":         {"role": {"enabled": False}},
    "output":       {"output": {"format": "plain_text", "strictness": "low",
                                "include_schema_in_prompt": False, "forbid_extra_fields": False}},
    "verification": {"verification": {"enabled": False}},
    "reasoning":    {"reasoning": {"strategy": "none", "visibility": "visible", "budget": "low"}},
    "layout":       {"layout": {"delimiter": "plain", "section_order": []}},
    "constraints":  {"constraints": {"placement": "top", "explicitness": "low", "max_count": 0}},
}


def variant_genome(base: PromptGenome, locus: str) -> PromptGenome:
    g = copy.deepcopy(base)
    for field, patch in ABLATIONS[locus].items():
        obj = getattr(g, field)
        for k, v in patch.items():
            setattr(obj, k, v)
    return g.model_validate(g.model_dump())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--emit", action="store_true", help="只打印各 variant 编译文本(零 API)")
    ap.add_argument("--loci", default="role,output,verification,reasoning,layout,constraints")
    ap.add_argument("--seed", type=int, default=924)
    ap.add_argument("--hold-n", type=int, default=8)
    ap.add_argument("--model", default="qwen")
    ap.add_argument("--timeout", type=float, default=600.0)
    ap.add_argument("--hard", type=float, default=640.0)
    args = ap.parse_args()

    spec = TaskSpec.from_yaml(str(REPO / "apc-pipeline" / "configs" / "tasks" / "external_math.yaml"))
    compiler = DefaultPromptCompiler()
    base = PromptGenome.from_json(str(REPO / "apc-pipeline" / "configs" / "genomes" / "base.json"))
    prof_cache = REPO / "artifacts" / "profiles" / "qwen_probe.json"
    if not prof_cache.exists():  # 探针缓存文件名= model_id_probe.json
        cands = list((REPO / "artifacts" / "profiles").glob("*probe*.json"))
        assert cands, "缺探针画像缓存(先跑过 probe 或 real_profile)"
        prof_cache = cands[0]
    profile = ModelProfile.model_validate_json(prof_cache.read_text(encoding="utf-8"))

    texts = {"base": H.compile_for(base, spec, profile, compiler)}
    for locus in ABLATIONS:
        texts[f"minus-{locus}"] = H.compile_for(variant_genome(base, locus), spec, profile, compiler)

    if args.emit:
        for name, t in texts.items():
            print(f"== {name}: {len(t)} chars ==")
            diff = sum(1 for a, b in zip(t, texts["base"]) if a != b)
            print(f"   ~diff-vs-base prefix-chars: {diff}")
        return 0

    H.G.JUDGE = H.HLEJudge()
    H.G.CHECKER = H.HLEChecker()
    H.G.document = lambda smp: smp["doc"]
    client = H.make_client(args.model, args.timeout)
    from apc.models.mock_client import MockClient
    if isinstance(client, MockClient):
        raise SystemExit("需要真实凭证")

    hold = H.load("holdout", args.hold_n)
    out_path = pathlib.Path(os.environ.get("HLE_ABL_OUT",
                                           str(REPO / "experiments/apcbench/real_hle_ablation.json")))
    doc = {"protocol": "hle-ablation-8q", "rows": []}
    if out_path.exists():
        doc = json.loads(out_path.read_text(encoding="utf-8"))
    for locus in [x.strip() for x in args.loci.split(",")]:
        name = f"minus-{locus}"
        t0 = time.time()
        b = RolloutBudget(10_000)
        sc, cases = H.run_eval(client, spec, hold, texts[name], b,
                               f"/tmp/hle_abl_{locus}_{args.seed}.jsonl",
                               hard=args.hard, timeout=args.timeout)
        acc = sum(1 for c in cases if float(c.get("accuracy", 0)) > .5)
        fmt = sum(1 for c in cases if float(c.get("format_score", 0)) > .5)
        row = {"task": "hle-abl", "seed": args.seed, "method": name,
               "holdout_score": round(sc, 4), "n_acc": acc, "n_format": fmt,
               "n": len(hold), "chars": len(texts[name]), "elapsed_s": round(time.time() - t0, 1)}
        doc["rows"] = [x for x in doc["rows"] if x["method"] != name] + [row]
        out_path.write_text(json.dumps(doc, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"{name:18s} hold={sc:.4f} acc={acc}/{len(hold)} fmt={fmt}/{len(hold)} "
              f"chars={len(texts[name])} ({row['elapsed_s']}s)", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
