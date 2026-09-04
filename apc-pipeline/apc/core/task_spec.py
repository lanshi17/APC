from __future__ import annotations
from typing import Any, Literal
from pydantic import BaseModel, Field

class OutputSpec(BaseModel):
    type: Literal["text", "json", "markdown", "table"] = "json"
    strict: bool = False
    schema_: dict[str, Any] | None = Field(default=None, alias="schema")
    model_config = {"populate_by_name": True}

class ReasoningSpec(BaseModel):
    required: bool = False
    visibility: Literal["visible", "hidden"] = "hidden"

class QualityWeights(BaseModel):
    accuracy: float = 0.5
    instruction_following: float = 0.2
    format: float = 0.15
    robustness: float = 0.1
    efficiency: float = 0.05

class CostLimits(BaseModel):
    max_input_tokens: int | None = None
    max_output_tokens: int | None = None
    max_latency_ms: int | None = None

class TaskSpec(BaseModel):
    task_id: str
    name: str
    version: str = "1.0.0"
    objective: str
    input: dict[str, Any] = Field(default_factory=dict)
    output: OutputSpec = Field(default_factory=OutputSpec)
    constraints: list[str] = Field(default_factory=list)
    reasoning: ReasoningSpec = Field(default_factory=ReasoningSpec)
    quality_weights: QualityWeights = Field(default_factory=QualityWeights)
    cost_limits: CostLimits = Field(default_factory=CostLimits)
    evaluation: dict[str, Any] = Field(default_factory=dict)

    @classmethod
    def from_yaml(cls, path: str) -> "TaskSpec":
        import yaml
        with open(path, encoding="utf-8") as f:
            data = yaml.safe_load(f)
        return cls.model_validate(data)
