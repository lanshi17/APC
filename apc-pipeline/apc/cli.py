"""APC CLI：task / probe / profile / prompt / eval / optimize / migrate。

约定：默认 Mock 模型（prefer_mock=True）保证无凭证可复现；--no-mock 走真实 API。
产物落盘 artifacts/{probes,profiles,prompts,evaluations,outputs,optimizations,migrations}/；
记录入库（SQLite，APC_DB_URL 或 data/apc.db）。
"""
from __future__ import annotations

import json
import pathlib
from datetime import datetime, timezone

import typer
from rich.console import Console
from rich.table import Table

app = typer.Typer(help="APC Adaptive Prompt Compiler")
console = Console()

task_app = typer.Typer(); app.add_typer(task_app, name="task")
probe_app = typer.Typer(); app.add_typer(probe_app, name="probe")
profile_app = typer.Typer(); app.add_typer(profile_app, name="profile")
prompt_app = typer.Typer(); app.add_typer(prompt_app, name="prompt")
eval_app = typer.Typer(); app.add_typer(eval_app, name="eval")
opt_app = typer.Typer(); app.add_typer(opt_app, name="optimize")
migrate_app = typer.Typer(); app.add_typer(migrate_app, name="migrate")

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
from dotenv import load_dotenv  # noqa: E402 （需在 _REPO_ROOT 之后定位 .env）

load_dotenv(_REPO_ROOT / ".env")
ARTIFACTS = _REPO_ROOT / "artifacts"
DATASETS = _REPO_ROOT / "datasets"


def _now():
    return datetime.now(timezone.utc).isoformat()


def _resolve_task(task: str) -> pathlib.Path:
    p = pathlib.Path(task)
    if p.exists():
        return p
    p = _REPO_ROOT / "configs" / "tasks" / f"{task}.yaml"
    if not p.exists():
        raise typer.BadParameter(f"找不到任务配置：{task}")
    return p


def _resolve_genome(genome: str) -> pathlib.Path:
    p = pathlib.Path(genome)
    if p.exists():
        return p
    p = _REPO_ROOT / "configs" / "genomes" / f"{genome}.json"
    if not p.exists():
        raise typer.BadParameter(f"找不到 genome 配置：{genome}")
    return p


def _load_profile(model: str):
    from apc.core.model_profile import ModelProfile

    pf = ARTIFACTS / "profiles" / f"{model}.json"
    if pf.exists():
        return ModelProfile.model_validate_json(pf.read_text(encoding="utf-8"))
    return ModelProfile(model_id=model)


def _save_json(path: pathlib.Path, payload) -> pathlib.Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def _client(model: str, mock: bool):
    from apc.models.factory import create_client

    return create_client(model, prefer_mock=mock)


def _session():
    from apc.storage.database import get_session

    return get_session()


def _probe_suite(suite: str) -> pathlib.Path:
    p = _REPO_ROOT / "probes" / suite
    if not p.exists():
        raise typer.BadParameter(f"找不到探针套件：{p}")
    return p


@task_app.command("validate")
def validate_task(config: str = typer.Option("configs/tasks/financial_analysis.yaml", "--config")):
    from apc.core.task_spec import TaskSpec

    spec = TaskSpec.from_yaml(_resolve_task(config))
    session = _session()
    from apc.storage import repository as repo

    repo.save_task(session, spec)
    console.print(f"[green]✓ Valid[/green] {spec.task_id} v{spec.version} — {spec.name}")
    console.print(f"  objective: {spec.objective}")
    console.print(f"  评测配置: {spec.evaluation}，权重: {spec.quality_weights}")


@probe_app.command("run")
def run_probe(model: str = typer.Option(..., "--model"),
              suite: str = typer.Option("v1", "--suite"),
              mock: bool = typer.Option(True, "--mock/--no-mock", help="默认 Mock，--no-mock 走真实 API")):
    from apc.models.factory import load_model_config
    from apc.profiler.probe_runner import ProbeRunner

    runner = ProbeRunner(_client(model, mock))
    results = runner.run_suite(_probe_suite(suite))
    out = _save_json(ARTIFACTS / "probes" / f"{model}.json",
                     {"model_id": model, "suite": suite, "generated_at": _now(),
                      "results": [r.model_dump(mode="json") if hasattr(r, "model_dump") else r
                                  for r in results]})
    table = Table(title=f"探针结果 {model}（{len(results)} 条）")
    table.add_column("probe_id"); table.add_column("category"); table.add_column("score")
    for r in results:
        rows = r if isinstance(r, dict) else r.model_dump()
        table.add_row(str(rows.get("probe_id", "")), str(rows.get("category", "")),
                      f"{float(rows.get('score', 0.0)):.2f}")
    console.print(table)
    console.print(f"[green]探针结果已落盘[/green] → {out}")


@profile_app.command("build")
def build_profile(model: str = typer.Option(..., "--model"),
                  suite: str = typer.Option("v1", "--suite")):
    """探针产物 → 15 维能力画像（无探针产物时现场补跑）。"""
    from apc.profiler.profile_builder import ProfileBuilder

    probe_file = ARTIFACTS / "probes" / f"{model}.json"
    if probe_file.exists():
        payload = json.loads(probe_file.read_text(encoding="utf-8"))
        probe_results = payload["results"]
    else:
        from apc.profiler.probe_runner import ProbeRunner

        probe_results = ProbeRunner(_client(model, True)).run_suite(_probe_suite(suite))
        probe_results = [r.model_dump(mode="json") if hasattr(r, "model_dump") else r
                         for r in probe_results]
    profile = ProfileBuilder().build(model, probe_results)
    out = _save_json(ARTIFACTS / "profiles" / f"{model}.json", profile.model_dump(mode="json"))

    session = _session()
    from apc.models.factory import load_model_config
    from apc.storage import repository as repo

    repo.save_model_profile(session, profile,
                            provider=load_model_config(model).get("provider", "unknown"))
    caps = profile.capability.model_dump()
    top = sorted(caps.items(), key=lambda kv: kv[1])[:3]
    console.print(f"[green]Profile written[/green] → {out}")
    console.print(f"  最弱三项: " + ", ".join(f"{k}={v:.2f}" for k, v in top))


@prompt_app.command("compile")
def compile_prompt(task: str = typer.Option(..., "--task"),
                   model: str = typer.Option(..., "--model"),
                   genome: str = typer.Option("configs/genomes/base.json", "--genome")):
    from apc.compiler.renderer import DefaultPromptCompiler
    from apc.core.genome import PromptGenome
    from apc.core.task_spec import TaskSpec

    spec = TaskSpec.from_yaml(_resolve_task(task))
    g = PromptGenome.from_json(_resolve_genome(genome))
    profile = _load_profile(model)
    prompt = DefaultPromptCompiler().compile(g, spec, profile)

    out_json = _save_json(ARTIFACTS / "prompts" / f"{spec.task_id}_{model}.json",
                          prompt.model_dump(mode="json"))
    out_txt = ARTIFACTS / "prompts" / f"{spec.task_id}_{model}.txt"
    out_txt.parent.mkdir(parents=True, exist_ok=True)
    out_txt.write_text(prompt.prompt_text, encoding="utf-8")

    session = _session()
    from apc.storage import repository as repo

    repo.save_genome(session, g, spec.task_id, is_baseline=True)
    repo.save_compiled_prompt(session, prompt)
    console.print(f"[green]Compiled[/green] → {out_txt.name} ({prompt.token_estimate} tokens, "
                  f"prompt_id={prompt.prompt_id}, genome_id={prompt.genome_id})")
    console.print(f"  JSON 产物: {out_json}")


def _compiled_prompt_from_artifact(path: pathlib.Path):
    from apc.core.prompt import CompiledPrompt

    if path.suffix == ".json":
        return CompiledPrompt.model_validate_json(path.read_text(encoding="utf-8"))
    raise typer.BadParameter(f"请传 prompt 的 JSON 产物（.json），不是 {path.suffix}")


@eval_app.command("run")
def run_eval(task: str = typer.Option(..., "--task"),
             model: str = typer.Option(..., "--model"),
             prompt: str = typer.Option(..., "--prompt"),
             dataset: str = typer.Option("dev", "--dataset"),
             mock: bool = typer.Option(True, "--mock/--no-mock"),
             judge: str = typer.Option("rule", "--judge", help="rule 或 llm:<model_id>")):
    """编译产物 → TrialResult（FR-5，KR-3 全字段追溯）。"""
    from apc.core.task_spec import TaskSpec
    from apc.evaluation.checker import RuleBasedChecker
    from apc.evaluation.dataset import load_dataset
    from apc.evaluation.judge import RuleBasedJudge, create_judge
    from apc.evaluation.runner import EvaluationRunner

    spec = TaskSpec.from_yaml(_resolve_task(task))
    cp = _compiled_prompt_from_artifact(_REPO_ROOT / prompt if not pathlib.Path(prompt).exists()
                                        else pathlib.Path(prompt))
    ds_file = DATASETS / "financial_analysis" / f"{dataset}.jsonl"
    ds = load_dataset(ds_file)
    judge_obj = RuleBasedJudge() if judge == "rule" else create_judge(judge, spec=spec)
    trial = EvaluationRunner(_client(model, mock), judge=judge_obj, checker=RuleBasedChecker(),
                             artifacts_dir=ARTIFACTS).evaluate(spec, cp, ds)

    out = _save_json(ARTIFACTS / "evaluations" / f"{trial.trial_id}.json", trial.model_dump(mode="json"))
    session = _session()
    from apc.storage import repository as repo

    repo.save_trial(session, trial, phase="eval")
    table = Table(title=f"TrialResult {trial.trial_id}")
    for k in ("score", "accuracy", "instruction_following", "format_score", "constraint_score",
              "robustness", "efficiency_score", "format_error_rate"):
        table.add_row(k, f"{getattr(trial, k):.4f}")
    console.print(table)
    console.print(f"[green]评测产物[/green] → {out}")
    console.print(f"  追溯: task={trial.task_id} model={trial.model_id} genome={trial.genome_id} "
                  f"prompt={trial.prompt_id} dataset={trial.dataset_id}@{trial.dataset_version}")


def _run_optimization(task_path: pathlib.Path, model: str, genome_path: pathlib.Path,
                      budget: int, generations: int | None = None,
                      population_size: int | None = None, seed: int = 42):
    """共享的优化执行器（optimize 命令与迁移共用）。"""
    import yaml
    from apc.compiler.renderer import DefaultPromptCompiler
    from apc.compiler.rules import CompilerRules
    from apc.core.genome import PromptGenome
    from apc.core.model_profile import ModelProfile
    from apc.core.task_spec import TaskSpec
    from apc.evaluation.checker import RuleBasedChecker
    from apc.evaluation.dataset import load_dataset
    from apc.evaluation.judge import RuleBasedJudge
    from apc.evaluation.runner import EvaluationRunner
    from apc.optimizer.evolutionary import EvolutionaryOptimizer

    spec = TaskSpec.from_yaml(task_path)
    profile = _load_profile(model)
    genome = PromptGenome.from_json(genome_path)
    compiler = DefaultPromptCompiler()
    dataset = load_dataset(DATASETS / "financial_analysis" / "dev.jsonl")
    runner = EvaluationRunner(_client(model, True), judge=RuleBasedJudge(),
                              checker=RuleBasedChecker(), artifacts_dir=ARTIFACTS)
    session = _session()
    from apc.storage import repository as repo

    repo.save_task(session, spec)

    def evaluate(g: PromptGenome, phase: str) -> float:
        # 每次评估把真实变异 genome 落库（KR-3 血统追溯：mutation_note + parent 链）
        cp = compiler.compile(g, spec, profile, apply_rules=False)
        repo.save_genome(session, g, spec.task_id, is_baseline=(g.parent_genome_id is None))
        return runner.evaluate(spec, cp, dataset, save_outputs=False).score

    cfg_path = _REPO_ROOT / "configs" / "optimizer" / "default.yaml"
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) if cfg_path.exists() else {}
    root = CompilerRules.apply(genome, profile)
    return EvolutionaryOptimizer(
        spec, profile, root, evaluate, seed=seed,
        generations=generations or cfg.get("generations", 3),
        population_size=population_size or cfg.get("population_size", 20),
        elite_k=cfg.get("elite_k", 5), budget=budget,
    ).optimize()


@opt_app.command("run")
def run_optimize(task: str = typer.Option(..., "--task"),
                 model: str = typer.Option(..., "--model"),
                 genome: str = typer.Option("configs/genomes/base.json", "--genome"),
                 budget: int = typer.Option(100, "--budget"),
                 seed: int = typer.Option(42, "--seed")):
    """进化搜索：变异只动 search_space，dev 淘汰 / validation 精英排序（FR-6）。"""
    from apc.core.genome import PromptGenome

    task_path = _resolve_task(task)
    genome_path = _resolve_genome(genome)
    report = _run_optimization(task_path, model, genome_path, budget, seed=seed)
    out = _save_json(ARTIFACTS / "optimizations" / f"{model}_report.json", report.model_dump(mode="json"))

    # 冠军 genome 落库（搜索过程的变异 genome 已在 evaluator 闭包中入库）
    session = _session()
    from apc.storage import repository as repo

    repo.save_genome(session, PromptGenome.model_validate(report.champion_genome), report.task_id)
    console.print(f"[green]优化完成[/green] {model}: baseline {report.baseline_score} → "
                  f"champion {report.champion_score}（提升 {report.improvement:+.4f}），"
                  f"预算 {report.budget_used}/{budget}，试验 {len(report.trials)} 次")
    for h in report.history:
        console.print(f"  第 {h['generation']} 代: best={h['best_score']:.4f} "
                      f"avg={h['avg_score']:.4f} budget={h['budget_used']}")
    console.print(f"[green]报告[/green] → {out}（champion_genome_id={report.champion_genome_id}）")


@migrate_app.command("run")
def run_migrate(task: str = typer.Option(..., "--task"),
                source_model: str = typer.Option(..., "--source-model"),
                target_model: str = typer.Option(..., "--target-model"),
                budget: int = typer.Option(30, "--budget"),
                seed: int = typer.Option(11, "--seed")):
    """CapabilityDelta → seed genome → 目标模型优化 → holdout 对比决策（FR-7 / KR-6）。"""
    from apc.compiler.renderer import DefaultPromptCompiler
    from apc.core.genome import PromptGenome
    from apc.core.task_spec import TaskSpec
    from apc.evaluation.checker import RuleBasedChecker
    from apc.evaluation.dataset import load_dataset
    from apc.evaluation.judge import RuleBasedJudge
    from apc.evaluation.runner import EvaluationRunner
    from apc.optimizer.evolutionary import EvolutionaryOptimizer
    from apc.transfer.migration import PromptMigrationPipeline

    task_path = _resolve_task(task)
    spec = TaskSpec.from_yaml(task_path)
    comp = DefaultPromptCompiler()
    dev = load_dataset(DATASETS / "financial_analysis" / "dev.jsonl")
    holdout = load_dataset(DATASETS / "financial_analysis" / "holdout.jsonl")
    profiles = {m: _load_profile(m) for m in (source_model, target_model)}

    def run_eval_on(genome: PromptGenome, mid: str, profile, dataset):
        cp = comp.compile(genome, spec, profile, apply_rules=False)
        return EvaluationRunner(_client(mid, True), judge=RuleBasedJudge(),
                                checker=RuleBasedChecker(), artifacts_dir=ARTIFACTS
                                ).evaluate(spec, cp, dataset, save_outputs=False)

    session = _session()
    from apc.storage import repository as repo

    def dev_eval(mid):
        def ev(g: PromptGenome, phase: str) -> float:
            # 迁移搜索过程的 genome 全部落库，保证 adapted 血统链完整（KR-3）
            if g.parent_genome_id is not None or g.mutation_note:
                repo.save_genome(session, g, spec.task_id)
            return run_eval_on(g, mid, profiles[mid], dev).score
        return ev

    def holdout_eval(g: PromptGenome, phase: str):
        mid = source_model if phase == "source_holdout" else target_model
        return run_eval_on(g, mid, profiles[mid], holdout)

    # 源冠军：优先复用 optimize 产物，否则用规则调整后的 base
    src_report_file = ARTIFACTS / "optimizations" / f"{source_model}_report.json"
    if src_report_file.exists():
        champ = PromptGenome.model_validate(
            json.loads(src_report_file.read_text(encoding="utf-8"))["champion_genome"])
        console.print(f"[dim]复用源冠军 {champ.genome_id}（来自 {src_report_file.name}）[/dim]")
    else:
        from apc.compiler.rules import CompilerRules

        champ = CompilerRules.apply(PromptGenome.from_json(
            _REPO_ROOT / "configs" / "genomes" / "base.json"), profiles[source_model])

    def optimizer_factory(seed_genome: PromptGenome):
        return EvolutionaryOptimizer(spec, profiles[target_model], seed_genome,
                                     dev_eval(target_model), seed=seed,
                                     generations=2, population_size=8, elite_k=3, budget=budget)

    pipeline = PromptMigrationPipeline(comp, dev_eval(target_model), holdout_eval, optimizer_factory)
    report = pipeline.migrate(spec, profiles[source_model], profiles[target_model], champ)

    out = _save_json(ARTIFACTS / "migrations" / f"{source_model}_to_{target_model}.json",
                     report.model_dump(mode="json"))
    repo.save_task(session, spec)
    repo.save_genome(session, PromptGenome.model_validate(report.adapted_genome), spec.task_id)
    color = "green" if report.decision == "adopt" else "yellow"
    console.print(f"[{color}]迁移决策: {report.decision}[/{color}]（KR-6: 目标 holdout "
                  f"{report.target_adapted_score} vs 源 {report.source_holdout_score}×0.9）")
    console.print(f"  能力差维度: {len(report.capability_delta)} | 预算 {report.budget_used}/{budget} "
                  f"| 格式错误率 {report.target_format_error_rate:.3f}")
    console.print(f"[green]报告[/green] → {out}")


if __name__ == "__main__":
    app()
