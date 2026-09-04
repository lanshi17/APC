from __future__ import annotations
from apc.core.model_profile import ModelProfile, CapabilityVector, BehaviorVector

class ProfileBuilder:
    def build(self, model_id: str, probe_results: list[dict]) -> ModelProfile:
        cap = CapabilityVector()
        # map category -> field
        cat_map = {
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
        for cat, field in cat_map.items():
            setattr(cap, field, self._avg(probe_results, cat))
        return ModelProfile(model_id=model_id, profile_version="v1", capability=cap, behavior=BehaviorVector(), raw_probe_results={"results": probe_results})

    def _avg(self, results: list[dict], category: str) -> float:
        vals = []
        for r in results:
            if r.get("category") == category:
                vs = [float(v) for v in r.get("metrics", {}).values() if isinstance(v, (bool,int,float))]
                if vs: vals.append(sum(vs)/len(vs))
        return sum(vals)/len(vals) if vals else 0.0
