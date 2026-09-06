# -*- coding: utf-8 -*-
"""可验证约束遵循基准（第四任务）：zero-shot / manual / random / apc-full / apc-pgam。

评测：RuleBasedChecker（通用 schema 检查）+ 本文件约束 Judge
（关键词命中/句数合规/无数字/置信度接近）+ TrialScorer 同权重。
画像/编译器/优化器与主管线同口径；预算 50，3 seeds。
输出 experiments/apcbench/follow_results.json。
"""
from __future__ import annotations

import json
import re
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
from apc.evaluation.scorer import TrialScorer
from apc.models.mock_client import MockClient
from apc.optimizer.evolutionary import EvolutionaryOptimizer, GenomeMutator, ProfileGuidedMutator
from apc.profiler.probe_runner import ProbeRunner
from apc.profiler.profile_builder import ProfileBuilder

TASK = REPO / "configs" / "tasks" / "constraint_following.yaml"
BASE_GENOME = REPO / "configs" / "genomes" / "base.json"
DS = REPO / "datasets" / "constraint_following"
MODELS = ["glm", "qwen", "gpt"]
SEEDS = [42, 43, 44]
BUDGET, GENS, POP, ELITE = 50, 3, 20, 5
METHODS = ["apc-pgam", "apc-full", "apc-no-profile", "apc-no-halving", "random-search", "zero-shot", "manual"]
_SENT = re.compile(r"[^。！？]+[。！？]")


def build_profile(model_id: str):
    results = ProbeRunner(MockClient(model_id)).run_suite(REPO / "probes" / "v1")
    dicts = [r if isinstance(r, dict) else r.model_dump(mode="json") for r in results]
    return ProfileBuilder().build(model_id, dicts)


def follow_case(expected: dict, output: str) -> dict:
    try:
        data = json.loads(output.strip())
        assert isinstance(data, dict)
    except Exception:
        return {"accuracy": 0.0, "instruction_following": 0.0, "output": output,
                "format_error": "json_parse_failed"}
    resp = str(data.get("response", ""))
    kw = str(expected.get("keyword", ""))
    try:
        max_sent = int(expected.get("max_sentences", 2))
    except (TypeError, ValueError):
        max_sent = 2
    kw_score = 1.0 if kw and kw in resp else 0.0
    n_sent = len(_SENT.findall(resp)) or (1 if resp.strip() else 0)
    sent_score = 1.0 if n_sent <= max_sent and n_sent > 0 else 0.0
    digit_score = 1.0 if not re.search(r"[0-9]", resp) else 0.0
    try:
        c_score = max(0.0, 1.0 - abs(float(expected.get("confidence", 0)) - float(data.get("confidence", 0))))
    except (TypeError, ValueError):
        c_score = 0.0
    acc = (kw_score + sent_score + digit_score + c_score) / 4
    conf = data.get("confidence")
    instr = (kw_score + sent_score + digit_score
             + (1.0 if isinstance(conf, (int, float)) and 0 <= conf <= 1 else 0.0)) / 4
    return {"accuracy": round(acc, 4), "instruction_following": round(instr, 4), "output": output}


def run_task(spec, compiler, profile, client, dataset, genome) -> object:
    import uuid
    checker = RuleBasedChecker()
    cp = compiler.compile(genome, spec, profile, apply_rules=False)
    cases = []
    for i, sample in enumerate(dataset.samples):
        doc = sample["input"]["document"]
        call = client.complete(cp.prompt_text.replace("{{input}}", doc))
        rule = checker.check(call.text, spec)
        cc = follow_case(sample.get("expected", {}), call.text)
        case = {"sample_id": f"{dataset.dataset_id}-{i}", "accuracy": cc["accuracy"],
                "instruction_following": cc["instruction_following"],
                "format_score": rule.get("format_score", 0.0),
                "constraint_score": rule.get("constraint_score", 0.0),
                "latency_ms": call.latency_ms, "input_tokens": call.input_tokens,
                "output_tokens": call.output_tokens, "output": call.text}
        if checker.format_error(rule):
            case["format_error"] = rule.get("format_error", "schema_violation")
        cases.append(case)
    return TrialScorer().score(spec, cases, trial_id=f"trial_{uuid.uuid4().hex[:12]}",
                               model_id=client.model_id, genome_id=genome.genome_id,
                               prompt_id=cp.prompt_id, dataset_id=dataset.dataset_id,
                               dataset_version=dataset.version, judge_id="follow_rule_v1",
                               temperature=0.0, model_version=client.model_version)


def _run_no_halving(spec, profile, root, ev, seed):
    """无 SHA 对照：每候选 dev_full 全量评估（预算口径一致），冠军选择与主管线一致。"""
    from copy import deepcopy as _dc
    from apc.optimizer.evolutionary import BudgetExhausted, OptimizationReport
    opt = EvolutionaryOptimizer(spec, profile, root, ev, generations=GENS,
                                population_size=POP, elite_k=ELITE,
                                budget=BUDGET, seed=seed)
    baseline = opt._eval(opt.root, "dev_full")
    population = opt._initial_population()
    best_g, best_s, hist = opt.root, baseline, []
    for gen in range(opt.generations):
        if opt.budget_used >= opt.budget:
            break
        try:
            scored = [(g, opt._eval(g, "dev_full")) for g in population]
        except BudgetExhausted:
            break
        scored.sort(key=lambda x: x[1], reverse=True)
        if scored and scored[0][1] > best_s:
            best_g, best_s = scored[0]
        elite = scored[: opt.elite_k] or [(best_g, best_s)]
        population = [_dc(g) for g, _ in elite]
        while len(population) < opt.population_size:
            mutated, _ = opt.mutator.mutate(opt.rng.choice(elite)[0])
            population.append(mutated)
        hist.append({"generation": gen + 1,
                     "best_score": round(max(s for _, s in scored), 4),
                     "avg_score": round(sum(s for _, s in scored) / len(scored), 4),
                     "budget_used": opt.budget_used, "survivors": len(scored)})
    val_scores: dict[str, float] = {}
    for g in (best_g, opt.root):
        try:
            val_scores[g.genome_id] = opt._eval(g, "validation")
        except BudgetExhausted:
            break
    if val_scores:
        champ_id = max(val_scores, key=lambda k: val_scores[k])
        champ = best_g if champ_id == best_g.genome_id else opt.root
        champ_s = val_scores[champ_id]
    else:
        champ, champ_s = best_g, best_s
    return OptimizationReport(task_id=spec.task_id, model_id=profile.model_id,
                              baseline_score=round(baseline, 4),
                              champion_score=round(champ_s, 4),
                              champion_genome=champ.model_dump(mode="json"),
                              champion_genome_id=champ.genome_id,
                              history=hist, trials=opt.trials, budget_used=opt.budget_used,
                              improvement=round(champ_s - baseline, 4))

def result_of(method: str, model_id: str, seed: int) -> dict:
    import random as _r
    spec = TaskSpec.from_yaml(TASK)
    base = PromptGenome.from_json(BASE_GENOME)
    compiler = DefaultPromptCompiler()
    profile = build_profile(model_id)
    client = MockClient(model_id)
    dev = load_dataset(DS / "dev.jsonl")
    val = load_dataset(DS / "validation.jsonl")
    hold = load_dataset(DS / "holdout.jsonl")
    t0 = time.time()

    def score_of(g, ds):
        return run_task(spec, compiler, profile, client, ds, g).score

    if method == "manual":
        g = deepcopy(base)
        g.examples.enabled, g.examples.count = True, 2
        g.output.strictness, g.output.include_schema_in_prompt = "high", True
        g.output.forbid_extra_fields, g.verification.enabled = True, True
        return pack(method, model_id, seed, score_of(g, dev), score_of(g, val),
                    score_of(g, hold), 0, t0)
    if method == "zero-shot":
        return pack(method, model_id, seed, score_of(base, dev), score_of(base, val),
                    score_of(base, hold), 0, t0)

    dev_r1 = dev.samples[:8]

    def ev(g: PromptGenome, phase: str) -> float:
        from apc.evaluation.dataset import Dataset
        ds = Dataset(dataset_id="dev_r1", version="f", samples=dev_r1) if phase == "dev_r1" \
            else (val if phase == "validation" else dev)
        return score_of(g, ds)

    root = CompilerRules.apply(deepcopy(base), profile) if method != "apc-no-profile" else deepcopy(base)
    if method == "random-search":
        mut = GenomeMutator(root.search_space or {}, _r.Random(seed))
        cands = [root]
        while len(cands) < BUDGET:
            m, note = mut.mutate(root)
            if note:
                cands.append(m)
        scored = sorted(((g, ev(g, "dev_full")) for g in cands), key=lambda x: x[1], reverse=True)
        top = sorted(((g, ev(g, "validation")) for g, _ in scored[:5]),
                     key=lambda x: x[1], reverse=True)
        champ, champ_v = top[0]
        return pack(method, model_id, seed, scored[0][1], champ_v,
                    score_of(champ, hold), len(cands), t0)
    if method == "apc-pgam":
        mut = ProfileGuidedMutator(root.search_space or {}, _r.Random(seed),
                                   profile.capability.model_dump())
        rep = EvolutionaryOptimizer(spec, profile, root, ev, generations=GENS,
                                    population_size=POP, elite_k=ELITE,
                                    budget=BUDGET, seed=seed, mutator=mut).optimize()
    elif method == "apc-no-halving":
        rep = _run_no_halving(spec, profile, root, ev, seed)
    else:
        rep = EvolutionaryOptimizer(spec, profile, root, ev, generations=GENS,
                                    population_size=POP, elite_k=ELITE,
                                    budget=BUDGET, seed=seed).optimize()
    champ = PromptGenome.model_validate(rep.champion_genome)
    return pack(method, model_id, seed, rep.baseline_score, score_of(champ, val),
                score_of(champ, hold), rep.budget_used, t0)


def pack(method, model_id, seed, baseline, validation, holdout, budget, t0):
    return {"method": method, "model_id": model_id, "seed": seed,
            "baseline_score": round(float(baseline), 4),
            "validation_score": round(float(validation), 4),
            "holdout_score": round(float(holdout), 4), "budget_used": budget,
            "elapsed_s": round(time.time() - t0, 1)}


def main():
    rows = []
    for method in METHODS:
        for model in MODELS:
            for seed in SEEDS:
                row = result_of(method, model, seed)
                rows.append(row)
                print(f"{method:15s} {model:5s} seed={seed} "
                      f"base={row['baseline_score']:.4f} hold={row['holdout_score']:.4f} "
                      f"budget={row['budget_used']}", flush=True)
    out = REPO / "experiments" / "apcbench" / "follow_results.json"
    out.write_text(json.dumps(rows, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n{len(rows)} runs -> {out}")


if __name__ == "__main__":
    main()
