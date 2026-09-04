from __future__ import annotations
import json, pathlib
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

@task_app.command("validate")
def validate_task(config: str = typer.Option(..., "--config")):
    from apc.core.task_spec import TaskSpec
    spec = TaskSpec.from_yaml(config)
    console.print(f"[green]✓ Valid[/green] {spec.task_id} v{spec.version} — {spec.name}")

@probe_app.command("run")
def run_probe(model: str = typer.Option(..., "--model"), suite: str = typer.Option("v1", "--suite")):
    from apc.models.mock_client import MockClient
    from apc.profiler.probe_runner import ProbeRunner
    import os
    suite_dir = f"probes/{suite}" if os.path.exists(f"probes/{suite}") else f"../probes/{suite}"
    runner = ProbeRunner(MockClient(model))
    results = runner.run_suite(suite_dir) if pathlib.Path(suite_dir).exists() else []
    console.print(f"[green]Probed {len(results)} cases[/green] for model={model}")
    for r in results[:5]: console.print(r)

@profile_app.command("build")
def build_profile(model: str = typer.Option(..., "--model")):
    from apc.profiler.profile_builder import ProfileBuilder
    builder = ProfileBuilder()
    profile = builder.build(model, [])
    out = f"artifacts/profiles/{model}.json"
    pathlib.Path(out).parent.mkdir(parents=True, exist_ok=True)
    pathlib.Path(out).write_text(profile.model_dump_json(indent=2), encoding="utf-8")
    console.print(f"[green]Profile written[/green] → {out}")

@prompt_app.command("compile")
def compile_prompt(task: str = typer.Option(..., "--task"), model: str = typer.Option(..., "--model"), genome: str = typer.Option(..., "--genome")):
    from apc.core.task_spec import TaskSpec
    from apc.core.genome import PromptGenome
    from apc.core.model_profile import ModelProfile
    from apc.compiler.renderer import DefaultPromptCompiler
    spec = TaskSpec.from_yaml(task)
    g = PromptGenome.from_json(genome)
    profile = ModelProfile(model_id=model)
    # try load profile file
    pf = pathlib.Path(f"artifacts/profiles/{model}.json")
    if pf.exists(): profile = ModelProfile.model_validate_json(pf.read_text(encoding="utf-8"))
    prompt = DefaultPromptCompiler().compile(g, spec, profile)
    out = f"artifacts/prompts/{spec.task_id}_{model}.txt"
    pathlib.Path(out).parent.mkdir(parents=True, exist_ok=True)
    pathlib.Path(out).write_text(prompt.prompt_text, encoding="utf-8")
    console.print(f"[green]Compiled[/green] → {out} ({prompt.token_estimate} tokens)")

@eval_app.command("run")
def run_eval(task: str = typer.Option(..., "--task"), model: str = typer.Option(..., "--model"), prompt: str = typer.Option(..., "--prompt"), dataset: str = typer.Option(..., "--dataset")):
    console.print(f"[green]Evaluating[/green] task={task} model={model} dataset={dataset}")
    console.print("[dim]RuleChecker + Judge → TrialResult (mock)[/dim]")

@opt_app.command("run")
def run_optimize(task: str = typer.Option(..., "--task"), model: str = typer.Option(..., "--model"), budget: int = typer.Option(100, "--budget")):
    from apc.graph.pipeline import graph
    # run LangGraph pipeline
    result = graph.invoke({"task_spec_path": f"configs/tasks/{task}.yaml", "model_id": model, "genome_path": "configs/genomes/base.json"})
    console.print(f"[green]Optimize done[/green] best_score={result.get('best_score')}")

@migrate_app.command("run")
def run_migrate(task: str = typer.Option(..., "--task"), source_model: str = typer.Option(..., "--source-model"), target_model: str = typer.Option(..., "--target-model")):
    console.print(f"[green]Migrating[/green] {task} {source_model} → {target_model}")

if __name__ == "__main__":
    app()
