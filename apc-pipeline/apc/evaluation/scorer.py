from __future__ import annotations

import statistics

from apc.core.task_spec import TaskSpec
from apc.core.trial import TrialResult


class TrialScorer:
    """按 PRD FR-5 固定公式合成最终得分（权重来自 TaskSpec）。"""

    def score(
        self,
        task_spec: TaskSpec,
        case_results: list[dict],
        *,
        trial_id: str,
        model_id: str,
        genome_id: str,
        prompt_id: str,
        dataset_id: str,
        dataset_version: str,
        judge_id: str,
        temperature: float = 0.0,
        model_version: str = "",
        parent_genome_id: str | None = None,
        mutation_note: str | None = None,
        robustness_override: float | None = None,
        metadata: dict | None = None,
    ) -> TrialResult:
        def avg(key: str) -> float:
            vals = [float(c[key]) for c in case_results if key in c and c[key] is not None]
            return sum(vals) / len(vals) if vals else 0.0

        accuracy = avg("accuracy")
        instruction_following = avg("instruction_following")
        format_score = avg("format_score")
        constraint_score = avg("constraint_score")
        format_errors = [c["sample_id"] for c in case_results if c.get("format_error")]

        if robustness_override is not None:
            robustness = robustness_override
        else:
            accs = [float(c.get("accuracy", 0.0)) for c in case_results]
            variance = statistics.pvariance(accs) if len(accs) > 1 else 0.0
            robustness = max(0.0, 1.0 - variance * 4)

        cl = (task_spec.cost_limits.model_dump() if task_spec.cost_limits else {})
        latencies = [float(c["latency_ms"]) for c in case_results if c.get("latency_ms") is not None]
        out_tokens = [float(c.get("output_tokens", 0)) for c in case_results]
        lat_score = 1.0
        tok_score = 1.0
        if cl.get("max_latency_ms") and latencies:
            lat_score = self._ratio_score(sum(latencies) / len(latencies), float(cl["max_latency_ms"]))
        if cl.get("max_output_tokens") and out_tokens:
            tok_score = self._ratio_score(sum(out_tokens) / len(out_tokens), float(cl["max_output_tokens"]))
        efficiency = round(0.5 * lat_score + 0.5 * tok_score, 4)

        w = task_spec.quality_weights
        total = (w.accuracy * accuracy + w.instruction_following * instruction_following
                 + w.format * format_score + w.robustness * robustness + w.efficiency * efficiency)

        return TrialResult(
            trial_id=trial_id,
            task_id=task_spec.task_id,
            model_id=model_id,
            model_version=model_version,
            genome_id=genome_id,
            parent_genome_id=parent_genome_id,
            mutation_note=mutation_note,
            prompt_id=prompt_id,
            dataset_id=dataset_id,
            dataset_version=dataset_version,
            judge_id=judge_id,
            score=round(total, 4),
            accuracy=round(accuracy, 4),
            instruction_following=round(instruction_following, 4),
            format_score=round(format_score, 4),
            constraint_score=round(constraint_score, 4),
            robustness=round(robustness, 4),
            efficiency_score=efficiency,
            case_results=case_results,
            format_error_rate=round(len(format_errors) / len(case_results), 4) if case_results else 0.0,
            avg_latency_ms=int(round(sum(latencies) / len(latencies))) if latencies else 0,
            total_input_tokens=int(sum(float(c.get("input_tokens", 0)) for c in case_results)),
            total_output_tokens=int(sum(out_tokens)),
            metadata={**(metadata or {}), "temperature": temperature},
        )

    @staticmethod
    def _ratio_score(actual: float, limit: float) -> float:
        if actual <= limit:
            return 1.0
        return max(0.0, 1.0 - (actual - limit) / limit)
