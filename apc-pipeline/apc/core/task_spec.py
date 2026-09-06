from __future__ import annotations
import re
from typing import Any, Literal
from pydantic import BaseModel, Field, model_validator


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
    examples: list[dict[str, Any]] = Field(default_factory=list)

    @model_validator(mode="after")
    def _validate(self) -> "TaskSpec":
        errors: list[str] = []
        if not self.task_id or not re.fullmatch(r"[a-z0-9_]+", self.task_id):
            errors.append(f"task_id 必须是非空的小写字母/数字/下划线，当前为 {self.task_id!r}")
        if not self.objective or not self.objective.strip():
            errors.append("objective 不能为空")
        if self.output.type == "json" and self.output.strict and not self.output.schema_:
            errors.append("严格 JSON 输出任务必须提供 output.schema（作为结构化输出的唯一来源）")
        total = self.quality_weights.accuracy + self.quality_weights.instruction_following + \
            self.quality_weights.format + self.quality_weights.robustness + self.quality_weights.efficiency
        if abs(total - 1.0) > 0.01:
            errors.append(f"quality_weights 权重之和必须为 1.0，当前为 {total:.4f}")
        for name in ("max_input_tokens", "max_output_tokens", "max_latency_ms"):
            v = getattr(self.cost_limits, name)
            if v is not None and v <= 0:
                errors.append(f"cost_limits.{name} 必须为正整数，当前为 {v}")
        if errors:
            raise ValueError("TaskSpec 校验失败：\n- " + "\n- ".join(errors))
        return self

    @classmethod
    def from_yaml(cls, path: str) -> "TaskSpec":
        import yaml
        with open(path, encoding="utf-8") as f:
            data = yaml.safe_load(f)
        try:
            return cls.model_validate(data)
        except ValueError as e:
            raise ValueError(f"{path}: {e}") from e
