"""LangGraph APC Pipeline — StateGraph 编排全流程（真实现，FR-1~FR-7 串接）。

节点：probe → build_profile → compile → evaluate → optimize。
所有节点读取/写回 APCState；模型默认 Mock（prefer_mock=True），真实凭证可用时由 CLI 层决定。
"""
from __future__ import annotations

import os
from copy import deepcopy
from pathlib import Path
from typing import Any, TypedDict

from langgraph.graph import END, StateGraph

_REPO_ROOT = Path(__file__).resolve().parents[3]
PROBE_SUITE_DIR = os.getenv("APC_PROBE_DIR", str(_REPO_ROOT / "probes" / "v1"))
DATA_DIR = _REPO_ROOT / "datasets" / "financial_analysis"
ARTIFACTS_DIR = _REPO_ROOT / "artifacts"


class APCState(TypedDict, total=False):
    task_spec_path: str
    genome_path: str
    model_id: str
    prefer_mock: bool
    budget: int
    generations: int
    population_size: int
    # 节点产出
    artifacts_dir: str
    probe_results: list[dict]
    profile: dict
    compiled_prompt: dict
    trial_result: dict
    optimization_report: dict


def _art(state: APCState) -> Path:
    """产物根目录：state.artifacts_dir 优先（测试隔离），否则仓库 artifacts/。"""
    return Path(state.get("artifacts_dir", str(ARTIFACTS_DIR)))


def node_probe(state: APCState) -> dict[str, Any]:
    """15 探针行为测量（FR-4）。"""
    from apc.models.factory import create_client
    from apc.profiler.probe_runner import ProbeRunner

    client = create_client(state["model_id"], prefer_mock=state.get("prefer_mock", True))
    results = ProbeRunner(client).run_suite(PROBE_SUITE_DIR)
    return {"probe_results": [
        r.model_dump(mode="json") if hasattr(r, "model_dump") else r for r in results]}


def node_build_profile(state: APCState) -> dict[str, Any]:
    """探针结果 → 15 维能力画像 + 7 维行为向量（FR-4）。"""
    from apc.core.model_profile import ModelProfile
    from apc.profiler.profile_builder import ProfileBuilder

    profile = ProfileBuilder().build(state["model_id"], state["probe_results"])
    # 落盘画像 + 供后续编译使用
    prof_dir = _art(state) / "profiles"
    prof_dir.mkdir(parents=True, exist_ok=True)
    profile_dict = profile.model_dump(mode="json")
    import json

    (prof_dir / f"{state['model_id']}.json").write_text(
        json.dumps(profile_dict, ensure_ascii=False, indent=2), encoding="utf-8")
    validated = ModelProfile.model_validate(profile_dict)
    return {"profile": validated.model_dump(mode="json")}


def node_compile(state: APCState) -> dict[str, Any]:
    """Genome + TaskSpec + ModelProfile → CompiledPrompt（FR-2）。"""
    import json

    from apc.compiler.renderer import DefaultPromptCompiler
    from apc.core.genome import PromptGenome
    from apc.core.model_profile import ModelProfile
    from apc.core.task_spec import TaskSpec

    spec = TaskSpec.from_yaml(state["task_spec_path"])
    genome = PromptGenome.from_json(state["genome_path"])
    profile = ModelProfile.model_validate(state["profile"])
    prompt = DefaultPromptCompiler().compile(genome, spec, profile)
    prompt_dir = _art(state) / "prompts"
    prompt_dir.mkdir(parents=True, exist_ok=True)
    (prompt_dir / f"{prompt.prompt_id}.json").write_text(
        json.dumps(prompt.model_dump(mode="json"), ensure_ascii=False, indent=2), encoding="utf-8")
    return {"compiled_prompt": prompt.model_dump(mode="json")}


def node_evaluate(state: APCState) -> dict[str, Any]:
    """基线评测：dev 数据集 + 规则 Judge + 加权评分（FR-5）。"""
    import json

    from apc.core.genome import PromptGenome
    from apc.core.model_profile import ModelProfile
    from apc.core.task_spec import TaskSpec
    from apc.evaluation.checker import RuleBasedChecker
    from apc.evaluation.dataset import load_dataset
    from apc.evaluation.judge import RuleBasedJudge
    from apc.evaluation.runner import EvaluationRunner
    from apc.models.factory import create_client

    spec = TaskSpec.from_yaml(state["task_spec_path"])
    prompt = json.loads(json.dumps(state["compiled_prompt"]))
    from apc.core.prompt import CompiledPrompt

    cp = CompiledPrompt.model_validate(prompt)
    genome = PromptGenome.from_json(state["genome_path"])
    client = create_client(state["model_id"], prefer_mock=state.get("prefer_mock", True))
    dataset = load_dataset(DATA_DIR / "dev.jsonl")
    trial = EvaluationRunner(client, judge=RuleBasedJudge(), checker=RuleBasedChecker(),
                             artifacts_dir=_art(state)).evaluate(spec, cp, dataset)
    return {"trial_result": trial.model_dump(mode="json")}


def node_optimize(state: APCState) -> dict[str, Any]:
    """进化优化：规则调整后的 root + apply_rules=False（FR-6）。"""
    import json

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
    from apc.models.factory import create_client
    from apc.optimizer.evolutionary import EvolutionaryOptimizer

    spec = TaskSpec.from_yaml(state["task_spec_path"])
    profile = ModelProfile.model_validate(state["profile"])
    genome = PromptGenome.from_json(state["genome_path"])
    client = create_client(state["model_id"], prefer_mock=state.get("prefer_mock", True))
    compiler = DefaultPromptCompiler()
    dataset = load_dataset(DATA_DIR / "dev.jsonl")
    runner = EvaluationRunner(client, judge=RuleBasedJudge(), checker=RuleBasedChecker(),
                              artifacts_dir=_art(state))

    def evaluate(g: PromptGenome, phase: str) -> float:
        cp = compiler.compile(g, spec, profile, apply_rules=False)
        result = runner.evaluate(spec, cp, dataset, save_outputs=False)
        return result.score

    cfg_path = _REPO_ROOT / "configs" / "optimizer" / "default.yaml"
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) if cfg_path.exists() else {}
    root = CompilerRules.apply(deepcopy(genome), profile)
    opt = EvolutionaryOptimizer(
        spec, profile, root, evaluate,
        generations=state.get("generations", cfg.get("generations", 3)),
        population_size=state.get("population_size", cfg.get("population_size", 20)),
        elite_k=state.get("elite_k", cfg.get("elite_k", 5)),
        budget=state.get("budget", cfg.get("budget", 100)),
    )
    report = opt.optimize()
    out_dir = _art(state) / "optimizations"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{state['model_id']}_report.json").write_text(
        json.dumps(report.model_dump(mode="json"), ensure_ascii=False, indent=2), encoding="utf-8")
    return {"optimization_report": report.model_dump(mode="json")}


def build_graph():
    g = StateGraph(APCState)
    g.add_node("probe", node_probe)
    g.add_node("build_profile", node_build_profile)
    g.add_node("compile", node_compile)
    g.add_node("evaluate", node_evaluate)
    g.add_node("optimize", node_optimize)
    g.set_entry_point("probe")
    g.add_edge("probe", "build_profile")
    g.add_edge("build_profile", "compile")
    g.add_edge("compile", "evaluate")
    g.add_edge("evaluate", "optimize")
    g.add_edge("optimize", END)
    return g.compile()


graph = build_graph()
