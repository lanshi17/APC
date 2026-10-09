# -*- coding: utf-8 -*-
"""APCBench: MPSO 顶会级基准(离线全 Mock,确定种子,多轮统计-bootstrap CI)。

方法(同一预算口径, Mock/dev):
- apc-full: Profile 驱动规则编译 + 进化搜索 + Successive Halving
- apc-no-profile: 不做 CompilerRules,搜索同配置(消融:画像价值)
- apc-no-halving: 每候选 dev_full 全量评估(消融:Halving 预算效率)
- random-search: 相同评估次数下纯随机变异(优化器价值)
- zero-shot: base genome 直接编译不动(下界)
- manual: base genome 手工启发式强化一次(genome+schema 全开,代表人工调优)

评估: dev 选型 → validation 模型选择 → holdout 独立报告(预算外)。
每方法 × 3 模型 × 5 seed; holdout 分数组内 bootstrap 95% CI(2000 次,另行脚本)；
跨方法对比另用配对 bootstrap 差值 CI 与 Holm 校正(analysis 脚本)。

输出 bench_results.json: 逐 run 记录(baseline/validation/holdout/budget)。
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
from apc.core.model_profile import ModelProfile
from apc.core.task_spec import TaskSpec
from apc.evaluation.checker import RuleBasedChecker
from apc.evaluation.dataset import load_dataset
from apc.evaluation.judge import RuleBasedJudge
from apc.evaluation.runner import EvaluationRunner
from apc.models.mock_client import MockClient
from apc.optimizer.evolutionary import EvolutionaryOptimizer
def build_profile(model_id: str) -> ModelProfile:
    """真实探针画像（与主管线同口径：ProbeRunner + ProfileBuilder）。"""
    from apc.profiler.probe_runner import ProbeRunner
    from apc.profiler.profile_builder import ProfileBuilder
    results = ProbeRunner(MockClient(model_id)).run_suite(REPO / "probes" / "v1")
    dicts = [r if isinstance(r, dict) else r.model_dump(mode="json") for r in results]
    return ProfileBuilder().build(model_id, dicts)
TASK = REPO / "configs" / "tasks" / "financial_analysis.yaml"
BASE_GENOME = REPO / "configs" / "genomes" / "base.json"
DS_DIR = REPO / "datasets" / "financial_analysis"
MODELS = ["glm", "qwen", "gpt"]
SEEDS = [42, 43, 44, 45, 46]
BUDGET = 100
GENS, POP, ELITE = 3, 20, 5


def build_profile(model_id: str) -> ModelProfile:
    return ModelProfile(model_id=model_id)


def make_runner(model_id: str, artifacts_dir: Path) -> EvaluationRunner:
    return EvaluationRunner(MockClient(model_id), judge=RuleBasedJudge(),
                            checker=RuleBasedChecker(), artifacts_dir=artifacts_dir)


def full_eval_factory(spec, compiler, profile, runner, dev_ds, holdout_ds):
    """evaluator(genome, phase): dev_r1=前8 dev, dev_full=dev 全量, validation, holdout 外部另算。"""
    dev_r1 = dev_ds.samples[:8]

    def ev(genome: PromptGenome, phase: str) -> float:
        cp = compiler.compile(genome, spec, profile, apply_rules=False)
        if phase == "dev_r1":
            from apc.evaluation.dataset import Dataset
            ds = Dataset(dataset_id="dev_r1", version="bench", samples=dev_r1)
        elif phase == "validation":
            ds = load_dataset(DS_DIR / "validation.jsonl")
        else:
            ds = dev_ds
        return runner.evaluate(spec, cp, ds, save_outputs=False).score

    def holdout(genome: PromptGenome) -> float:
        cp = compiler.compile(genome, spec, profile, apply_rules=False)
        return runner.evaluate(spec, cp, holdout_ds, save_outputs=False).score

    return ev, holdout


def run_optimizer(spec, profile, root, ev, seed, *, no_halving=False):
    if not no_halving:
        return EvolutionaryOptimizer(spec, profile, root, ev, generations=GENS,
                                     population_size=POP, elite_k=ELITE,
                                     budget=BUDGET, seed=seed).optimize()
    # no-halving 变体: 所有候选 dev_full 全量评估(同 budget 耗尽即停)
    from apc.optimizer.evolutionary import BudgetExhausted
    opt = EvolutionaryOptimizer(spec, profile, root, ev, generations=GENS,
                                population_size=POP, elite_k=ELITE,
                                budget=BUDGET, seed=seed)

    class _NH(EvolutionaryOptimizer):
        def optimize(self):
            baseline = self._eval(self.root, "dev_full")
            population = self._initial_population()
            best_g, best_s, hist = self.root, baseline, []
            for gen in range(self.generations):
                if self.budget_used >= self.budget:
                    break
                try:
                    scored = [(g, self._eval(g, "dev_full")) for g in population]
                except BudgetExhausted:
                    break
                scored.sort(key=lambda x: x[1], reverse=True)
                if scored and scored[0][1] > best_s:
                    best_g, best_s = scored[0]
                elite = scored[: self.elite_k] or [(best_g, best_s)]
                population = [deepcopy(g) for g, _ in elite]
                while len(population) < self.population_size:
                    mutated, _ = self.mutator.mutate(self.rng.choice(elite)[0])
                    population.append(mutated)
                hist.append({"generation": gen + 1,
                             "best_score": round(max(s for _, s in scored), 4),
                             "avg_score": round(sum(s for _, s in scored) / len(scored), 4),
                             "budget_used": self.budget_used,
                             "survivors": len(scored)})
            # 冠军选择与主管线一致：validation 可用时按其选，否则回退 dev 最优
            # （原先预算耗尽时默认 root，会掩盖已找到的 best）。
            val_scores: dict[str, float] = {}
            for g in (best_g, self.root):
                try:
                    val_scores[g.genome_id] = self._eval(g, "validation")
                except BudgetExhausted:
                    break
            if val_scores:
                champ_id = max(val_scores, key=lambda k: val_scores[k])
                champ = best_g if champ_id == best_g.genome_id else self.root
                champ_s = val_scores[champ_id]
            else:
                champ, champ_s = best_g, best_s
            from apc.optimizer.evolutionary import OptimizationReport
            return OptimizationReport(task_id=self.task_spec.task_id, model_id=self.model_profile.model_id,
                                      baseline_score=round(baseline, 4),
                                      champion_score=round(champ_s if champ_s is not None else best_s, 4),
                                      champion_genome=champ.model_dump(mode="json"),
                                      champion_genome_id=champ.genome_id,
                                      history=hist, trials=self.trials, budget_used=self.budget_used,
                                      improvement=round((champ_s if champ_s is not None else best_s) - baseline, 4))

    return _NH(spec, profile, root, ev, generations=GENS, population_size=POP,
               elite_k=ELITE, budget=BUDGET, seed=seed).optimize()


def result_of(method: str, model_id: str, seed: int, out_root: Path) -> dict:
    spec = TaskSpec.from_yaml(TASK)
    base = PromptGenome.from_json(BASE_GENOME)
    compiler = DefaultPromptCompiler()
    profile = build_profile(model_id)
    dev_ds = load_dataset(DS_DIR / "dev.jsonl")
    val_ds = load_dataset(DS_DIR / "validation.jsonl")
    holdout_ds = load_dataset(DS_DIR / "holdout.jsonl")
    runner = make_runner(model_id, out_root)
    t0 = time.time()

    if method == "manual":
        # 人工启发式强化: 全开 schema/校验/示例(不耗预算,单次评估)
        g = deepcopy(base)
        g.examples.enabled, g.examples.count = True, 2
        g.output.strictness, g.output.include_schema_in_prompt = "high", True
        g.output.forbid_extra_fields, g.verification.enabled = True, True
        cp = compiler.compile(g, spec, profile, apply_rules=False)
        b = runner.evaluate(spec, cp, dev_ds, save_outputs=False).score
        v = runner.evaluate(spec, cp, val_ds, save_outputs=False).score
        h = runner.evaluate(spec, cp, holdout_ds, save_outputs=False).score
        return pack(method, model_id, seed, b, v, h, 0, t0,
                    champion_genome=g.model_dump(mode="json"))

    if method == "zero-shot":
        cp = compiler.compile(base, spec, profile, apply_rules=False)
        b = runner.evaluate(spec, cp, dev_ds, save_outputs=False).score
        v = runner.evaluate(spec, cp, val_ds, save_outputs=False).score
        h = runner.evaluate(spec, cp, holdout_ds, save_outputs=False).score
        return pack(method, model_id, seed, b, v, h, 0, t0,
                    champion_genome=base.model_dump(mode="json"))

    ev, _ = full_eval_factory(spec, compiler, profile, runner, dev_ds, holdout_ds)

    if method in ("apc-full", "apc-no-profile", "apc-no-halving", "apc-pgam", "random-search"):
        need_rules = method not in ("apc-no-profile",)
        root = CompilerRules.apply(deepcopy(base), profile) if need_rules else deepcopy(base)
        if method == "random-search":
            # 随机搜索: 同 evaluate 次数(BUDGET),纯随机变异选 dev 最优,再 validation 冠军选择
            from apc.optimizer.evolutionary import GenomeMutator
            import random as _r
            rng = _r.Random(seed)
            mut = GenomeMutator(root.search_space or {}, rng)
            cands = [root]
            while len(cands) < BUDGET:
                m, note = mut.mutate(root)
                if note:
                    cands.append(m)
            scored = sorted(((g, ev(g, "dev_full")) for g in cands),
                            key=lambda x: x[1], reverse=True)
            budget_used = len(cands)
            top = scored[:5]
            valv = sorted(((g, ev(g, "validation")) for g, _ in top),
                          key=lambda x: x[1], reverse=True)
            champ, champ_v = valv[0]
            base_dev = ev(deepcopy(root), "dev_full") if budget_used < BUDGET else scored[-1][1]
            rep = {"trials": [], "budget_used": budget_used}
            hold = holdout_score(spec, compiler, profile, runner, holdout_ds, champ)
            return pack(method, model_id, seed, scored[0][1], champ_v, hold, budget_used, t0,
                        champion=champ.genome_id, champion_genome=champ.model_dump(mode="json"))
        if method == "apc-pgam":
            import random as _r
            from apc.optimizer.evolutionary import ProfileGuidedMutator
            pg_mut = ProfileGuidedMutator(root.search_space or {},
                                          _r.Random(seed),
                                          profile.capability.model_dump())
            rep = EvolutionaryOptimizer(spec, profile, root, ev, generations=GENS,
                                        population_size=POP, elite_k=ELITE,
                                        budget=BUDGET, seed=seed, mutator=pg_mut).optimize()
        else:
            rep = run_optimizer(spec, profile, root, ev, seed, no_halving=(method == "apc-no-halving"))
        champ = PromptGenome.model_validate(rep.champion_genome)
        hold = holdout_score(spec, compiler, profile, runner, holdout_ds, champ)
        val_hold = runner.evaluate(spec, compiler.compile(champ, spec, profile, apply_rules=False),
                                   val_ds, save_outputs=False).score
        return pack(method, model_id, seed, rep.baseline_score, val_hold, hold, rep.budget_used, t0,
                    champion=rep.champion_genome_id, history=rep.history,
                    champion_genome=rep.champion_genome)
    raise ValueError(method)


def holdout_score(spec, compiler, profile, runner, holdout_ds, genome):
    cp = compiler.compile(genome, spec, profile, apply_rules=False)
    return runner.evaluate(spec, cp, holdout_ds, save_outputs=False).score


def pack(method, model_id, seed, baseline, validation, holdout, budget, t0, champion="",
         history=None, champion_genome=None):
    return {"method": method, "model_id": model_id, "seed": seed,
            "baseline_score": round(float(baseline), 4), "validation_score": round(float(validation), 4),
            "holdout_score": round(float(holdout), 4), "budget_used": budget,
            "champion_genome_id": champion, "history": history or [],
            "champion_genome": champion_genome,
            "elapsed_s": round(time.time() - t0, 1)}

def main():
    global BUDGET
    if len(sys.argv) > 1:
        BUDGET = int(sys.argv[1])
    out = REPO / "experiments" / "apcbench"
    out.mkdir(parents=True, exist_ok=True)
    methods = ["apc-pgam", "apc-full", "apc-no-profile", "apc-no-halving", "random-search", "zero-shot", "manual"]
    rows = []
    for method in methods:
        for model in MODELS:
            for seed in SEEDS:
                row = result_of(method, model, seed, out / "tmp_artifacts")
                rows.append(row)
                print(f"{method:15s} {model:5s} seed={seed} "
                      f"base={row['baseline_score']:.4f} val={row['validation_score']:.4f} "
                      f"hold={row['holdout_score']:.4f} budget={row['budget_used']}", flush=True)
    fname = f"bench_results_b{BUDGET}.json"
    (out / fname).write_text(json.dumps(rows, ensure_ascii=False, indent=1), encoding="utf-8")
    (out / "bench_results.json").write_text(json.dumps(rows, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n{len(rows)} runs -> {out / fname} (also bench_results.json)")


if __name__ == "__main__":
    main()
