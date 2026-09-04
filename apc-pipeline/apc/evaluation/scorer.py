from __future__ import annotations
from apc.core.task_spec import TaskSpec
from apc.core.trial import TrialResult

class TrialScorer:
    def score(self, task_spec: TaskSpec, rule_results: list[dict], judge_results: list[dict], latency_ms_list: list[int], token_list: list[int]) -> TrialResult:
        def avg(vs): return sum(vs)/len(vs) if vs else 0.0
        accuracy = avg([r.get("accuracy",0) for r in judge_results])
        constraint = avg([r.get("constraint_following",0) for r in judge_results])
        format_score = avg([r.get("format_score",0) for r in rule_results])
        # robustness = 1 - variance
        scores = [r.get("accuracy",0) for r in judge_results]
        var = self._var(scores)
        robustness = max(0.0, 1.0-var)
        w = task_spec.quality_weights
        total = w.accuracy*accuracy + w.instruction_following*constraint + w.format*format_score + w.robustness*robustness + w.efficiency*self._eff(latency_ms_list, token_list, task_spec)
        return TrialResult(
            trial_id="trial", task_id=task_spec.task_id, model_id="model", genome_id="genome", prompt_id="prompt", dataset_id="dataset",
            score=total, accuracy=accuracy, instruction_following=constraint, format_score=format_score, constraint_score=constraint,
            robustness=robustness, efficiency_score=self._eff(latency_ms_list, token_list, task_spec),
            total_input_tokens=sum(token_list), total_output_tokens=0, latency_ms=int(avg(latency_ms_list)), variance=var, case_results=[]
        )
    def _var(self, vs: list[float]) -> float:
        if not vs: return 0
        m = sum(vs)/len(vs)
        return sum((x-m)**2 for x in vs)/len(vs)
    def _eff(self, lat, toks, spec):
        avg_lat = sum(lat)/len(lat) if lat else 0
        avg_tok = sum(toks)/len(toks) if toks else 0
        ls = max(0.0, 1.0 - avg_lat/spec.cost_limits.max_latency_ms) if spec.cost_limits.max_latency_ms else 1.0
        ts = max(0.0, 1.0 - avg_tok/spec.cost_limits.max_input_tokens) if spec.cost_limits.max_input_tokens else 1.0
        return 0.5*ls+0.5*ts
