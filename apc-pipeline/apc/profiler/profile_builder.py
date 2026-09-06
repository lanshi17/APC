from __future__ import annotations

from apc.core.model_profile import BehaviorVector, CapabilityVector, ModelProfile
from apc.profiler.probe_runner import BehaviorAnalyzer

_CATEGORY_FIELD = {
    "instruction_following": "instruction_following",
    "multi_constraint_following": "multi_constraint_following",
    "long_context": "long_context",
    "json_reliability": "json_reliability",
    "schema_strictness": "schema_strictness",
    "table_understanding": "table_understanding",
    "math": "math",
    "reasoning": "reasoning",
    "information_extraction": "information_extraction",
    "few_shot_benefit": "few_shot_benefit",
    "self_verification_benefit": "self_verification_benefit",
    "tool_usage": "tool_usage",
    "robustness_to_distraction": "robustness_to_distraction",
    "chinese_semantic": "chinese_semantic",
    "safety_boundary": "safety_boundary",
}


class ProfileBuilder:
    """探针原始结果 → CapabilityVector[15] + BehaviorVector[7] + 原始证据（FR-4）。"""

    def build(self, model_id: str, probe_results: list[dict], profile_version: str = "v1") -> ModelProfile:
        cap = CapabilityVector()
        for cat, field in _CATEGORY_FIELD.items():
            setattr(cap, field, round(self._avg_score(probe_results, cat), 4))
        behavior = BehaviorVector(**BehaviorAnalyzer.analyze(probe_results))
        return ModelProfile(
            model_id=model_id,
            profile_version=profile_version,
            capability=cap,
            behavior=behavior,
            raw_probe_results={"results": probe_results},
        )

    @staticmethod
    def _avg_score(results: list[dict], category: str) -> float:
        scores = [float(r["score"]) for r in results if r.get("category") == category]
        return sum(scores) / len(scores) if scores else 0.0
