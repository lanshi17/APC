# -*- coding: utf-8 -*-
"""迁移基准: 直接迁移 vs APC 迁移协议 vs 目标原生优化(上界)。

pairs: glm→qwen, glm→gpt, qwen→gpt (+反向 qwen→glm)。
每对 × 3 seed:
1. source 上全预算优化(100)得冠军 Cs;记 source_holdout(Cs @ source)。
2. direct: holdout(Cs @ target)。
3. adapted: PromptMigrationPipeline(CompilerRules+MigrationMutator seed,
   目标上小预算30重优化)→ holdout(Ca @ target)。
4. native: target 上全预算优化(100)冠军 Cn → holdout(Cn @ target,上界)。
指标: decay_direct = direct/native, recover = adapted/native, KR-6: adapted ≥ 0.9×source。
画像: 真实探针画像(与主管线同口径)。
"""
from __future__ import annotations

import json
import sys
import time
from copy import deepcopy
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "apc-pipeline"))

from apc.compiler.renderer import DefaultPromptCompiler
from apc.compiler.rules import CompilerRules
from apc.core.genome import PromptGenome
from apc.core.task_spec import TaskSpec
from apc.evaluation.checker import RuleBasedChecker
from apc.evaluation.dataset import load_dataset
from apc.evaluation.judge import RuleBasedJudge
from apc.evaluation.runner import EvaluationRunner
from apc.models.mock_client import MockClient
from apc.optimizer.evolutionary import EvolutionaryOptimizer
from apc.profiler.probe_runner import ProbeRunner
from apc.profiler.profile_builder import ProfileBuilder
from apc.transfer.migration import PromptMigrationPipeline

TASK = REPO / "configs" / "tasks" / "financial_analysis.yaml"
BASE_GENOME = REPO / "configs" / "genomes" / "base.json"
DS = REPO / "datasets" / "financial_analysis"
PAIRS = [("glm", "qwen"), ("glm", "gpt"), ("qwen", "gpt"), ("qwen", "glm")]
SEEDS = [42, 43, 44]
SRC_BUDGET, ADAPT_BUDGET = 100, 30


def build_profile(model_id: str):
    results = ProbeRunner(MockClient(model_id)).run_suite(REPO / "probes" / "v1")
    dicts = [r if isinstance(r, dict) else r.model_dump(mode="json") for r in results]
    return ProfileBuilder().build(model_id, dicts)


def optimize_on(spec, profile, model_id, dev, seed, budget, gens=3, pop=20):
    comp = DefaultPromptCompiler()
    base = PromptGenome.from_json(BASE_GENOME)
    runner = EvaluationRunner(MockClient(model_id), judge=RuleBasedJudge(),
                              checker=RuleBasedChecker(), artifacts_dir=Path("/tmp/bm-t"))
    dev_r1 = dev.samples[:8]

    def ev(g: PromptGenome, phase: str) -> float:
        cp = comp.compile(g, spec, profile, apply_rules=False)
        from apc.evaluation.dataset import Dataset
        if phase == "dev_r1":
            ds = Dataset(dataset_id="dev_r1", version="t", samples=dev_r1)
        else:
            ds = dev
        return runner.evaluate(spec, cp, ds, save_outputs=False).score

    root = CompilerRules.apply(deepcopy(base), profile)
    rep = EvolutionaryOptimizer(spec, profile, root, ev, generations=gens,
                                population_size=pop, elite_k=5,
                                budget=budget, seed=seed).optimize()
    return PromptGenome.model_validate(rep.champion_genome)


def holdout_score(spec, profile, model_id, genome, holdout) -> float:
    comp = DefaultPromptCompiler()
    cp = comp.compile(genome, spec, profile, apply_rules=False)
    return EvaluationRunner(MockClient(model_id), judge=RuleBasedJudge(),
                            checker=RuleBasedChecker(),
                            artifacts_dir=Path("/tmp/bm-t")).evaluate(
        spec, cp, holdout, save_outputs=False).score


def main():
    spec = TaskSpec.from_yaml(TASK)
    dev = load_dataset(DS / "dev.jsonl")
    holdout = load_dataset(DS / "holdout.jsonl")
    profiles = {m: build_profile(m) for m in ("glm", "qwen", "gpt")}
    comp = DefaultPromptCompiler()
    rows = []
    for src, tgt in PAIRS:
        for seed in SEEDS:
            t0 = time.time()
            cs = optimize_on(spec, profiles[src], src, dev, seed, SRC_BUDGET)
            src_hold = holdout_score(spec, profiles[src], src, cs, holdout)
            direct = holdout_score(spec, profiles[tgt], tgt, cs, holdout)

            # APC 迁移协议
            from apc.transfer.migration import CapabilityDeltaCalculator
            tgt_runner_dev = None

            def dev_eval(g: PromptGenome, phase: str) -> float:
                cp = comp.compile(g, spec, profiles[tgt], apply_rules=False)
                from apc.evaluation.dataset import Dataset
                ds = Dataset(dataset_id="d", version="t",
                             samples=dev.samples[:8]) if phase == "dev_r1" else dev
                return EvaluationRunner(MockClient(tgt), judge=RuleBasedJudge(),
                                        checker=RuleBasedChecker(),
                                        artifacts_dir=Path("/tmp/bm-t")).evaluate(
                    spec, cp, ds, save_outputs=False).score

            def hold_eval(g: PromptGenome, phase: str):
                mid = src if phase == "source_holdout" else tgt
                prof = profiles[mid]

                class _T:
                    pass
                t = _T()
                t.score = holdout_score(spec, prof, mid, g, holdout)
                cps = comp.compile(g, spec, prof, apply_rules=False)
                tr = EvaluationRunner(MockClient(mid), judge=RuleBasedJudge(),
                                      checker=RuleBasedChecker(),
                                      artifacts_dir=Path("/tmp/bm-t")).evaluate(
                    spec, cps, holdout, save_outputs=False)
                t.format_error_rate = tr.format_error_rate
                t.avg_latency_ms = tr.avg_latency_ms
                t.total_output_tokens = tr.total_output_tokens
                return t

            def opt_factory(seed_genome: PromptGenome):
                return EvolutionaryOptimizer(spec, profiles[tgt], seed_genome,
                                             dev_eval, seed=seed, generations=2,
                                             population_size=8, elite_k=3,
                                             budget=ADAPT_BUDGET)

            pipe = PromptMigrationPipeline(comp, dev_eval, hold_eval, opt_factory)
            rep = pipe.migrate(spec, profiles[src], profiles[tgt], cs)
            adapted = PromptGenome.model_validate(rep.adapted_genome)
            native = optimize_on(spec, profiles[tgt], tgt, dev, seed, SRC_BUDGET)
            native_hold = holdout_score(spec, profiles[tgt], tgt, native, holdout)
            row = {"source": src, "target": tgt, "seed": seed,
                   "source_holdout": round(src_hold, 4), "direct": round(direct, 4),
                   "adapted": round(rep.target_adapted_score, 4),
                   "native": round(native_hold, 4),
                   "decision": rep.decision,
                   "decay_direct": round(direct / native_hold, 4) if native_hold else None,
                   "recover": round(rep.target_adapted_score / native_hold, 4) if native_hold else None,
                   "kr6": bool(rep.target_adapted_score >= 0.9 * src_hold),
                   "elapsed_s": round(time.time() - t0, 1)}
            rows.append(row)
            print(f"{src}->{tgt} seed={seed} src={src_hold:.4f} direct={direct:.4f} "
                  f"adapted={rep.target_adapted_score:.4f} native={native_hold:.4f} "
                  f"recover={row['recover']:.4f} KR6={row['kr6']} {rep.decision}", flush=True)
    out = REPO / "experiments" / "apcbench" / "transfer_results.json"
    out.write_text(json.dumps(rows, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n{len(rows)} transfers -> {out}")


if __name__ == "__main__":
    main()
