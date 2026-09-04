"""LangGraph APC Pipeline — StateGraph 编排全流程。"""
from __future__ import annotations
from typing import TypedDict, Any
from langgraph.graph import StateGraph, END

class APCState(TypedDict, total=False):
    task_spec_path: str
    model_id: str
    genome_path: str
    probe_results: list[dict]
    profile: dict
    compiled_prompt: str
    trial_result: dict
    best_genome: dict
    best_score: float

# 节点：空实现骨架，实际调用 profiler/compiler/evaluation/optimizer
def node_probe(state: APCState) -> APCState:
    from apc.models.mock_client import MockClient
    from apc.profiler.probe_runner import ProbeRunner
    import os
    suite = os.path.join(os.path.dirname(__file__), "../../../probes/v1")
    # fallback if not found
    runner = ProbeRunner(MockClient(state.get("model_id","mock")))
    try:
        results = runner.run_suite(suite) if os.path.exists(suite) else []
    except Exception: results = []
    return {"probe_results": results}

def node_build_profile(state: APCState) -> APCState:
    from apc.profiler.profile_builder import ProfileBuilder
    builder = ProfileBuilder()
    profile = builder.build(state.get("model_id","mock"), state.get("probe_results",[]))
    return {"profile": profile.model_dump()}

def node_compile(state: APCState) -> APCState:
    from apc.core.task_spec import TaskSpec
    from apc.core.genome import PromptGenome
    from apc.core.model_profile import ModelProfile
    from apc.compiler.renderer import DefaultPromptCompiler
    try:
        spec = TaskSpec.from_yaml(state["task_spec_path"])
        genome = PromptGenome.from_json(state["genome_path"])
        profile = ModelProfile.model_validate(state["profile"])
        prompt = DefaultPromptCompiler().compile(genome, spec, profile)
        return {"compiled_prompt": prompt.prompt_text}
    except Exception as e:
        return {"compiled_prompt": f"compile_error: {e}"}

def node_evaluate(state: APCState) -> APCState:
    # placeholder: 实际评测需 dataset + checker + judge
    return {"trial_result": {"score": 0.85}}

def node_optimize(state: APCState) -> APCState:
    from apc.core.task_spec import TaskSpec
    from apc.core.genome import PromptGenome
    from apc.core.model_profile import ModelProfile
    from apc.compiler.renderer import DefaultPromptCompiler
    from apc.optimizer.evolutionary import EvolutionaryOptimizer
    class DummyEval:
        def evaluate_prompt(self, p): return 0.9
    try:
        spec = TaskSpec.from_yaml(state["task_spec_path"])
        genome = PromptGenome.from_json(state["genome_path"])
        profile = ModelProfile.model_validate(state["profile"])
        compiler = DefaultPromptCompiler()
        opt = EvolutionaryOptimizer(compiler, DummyEval())
        best, score = opt.optimize(spec, profile, [genome], generations=2, population_size=10)
        return {"best_genome": best.model_dump(), "best_score": score}
    except Exception as e:
        return {"best_genome": {}, "best_score": 0.0}

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
