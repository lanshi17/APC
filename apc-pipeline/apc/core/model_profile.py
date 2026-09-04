from __future__ import annotations
from typing import Any, Literal
from pydantic import BaseModel, Field

class CapabilityVector(BaseModel):
    instruction_following: float = 0.0
    multi_constraint_following: float = 0.0
    long_context: float = 0.0
    json_reliability: float = 0.0
    schema_strictness: float = 0.0
    table_understanding: float = 0.0
    math: float = 0.0
    reasoning: float = 0.0
    information_extraction: float = 0.0
    few_shot_benefit: float = 0.0
    self_verification_benefit: float = 0.0
    tool_usage: float = 0.0
    robustness_to_distraction: float = 0.0
    chinese_semantic: float = 0.0
    safety_boundary: float = 0.0

class BehaviorVector(BaseModel):
    verbosity: Literal["low","medium","high"] = "medium"
    prefix_tendency: Literal["low","medium","high"] = "low"
    json_prefix_noise: Literal["low","medium","high"] = "low"
    over_refusal: Literal["low","medium","high"] = "low"
    hallucination_tendency: Literal["low","medium","high"] = "low"
    latency: Literal["low","medium","high"] = "medium"
    cost: Literal["low","medium","high"] = "medium"

class ModelProfile(BaseModel):
    model_id: str
    profile_version: str = "v1"
    capability: CapabilityVector = Field(default_factory=CapabilityVector)
    behavior: BehaviorVector = Field(default_factory=BehaviorVector)
    raw_probe_results: dict[str, Any] = Field(default_factory=dict)
