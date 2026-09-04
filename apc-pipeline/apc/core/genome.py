from __future__ import annotations
from typing import Any, Literal
from pydantic import BaseModel, Field

class RoleGene(BaseModel):
    enabled: bool = False
    style: Literal["expert","assistant","auditor","analyst"] = "expert"
    authority_level: Literal["low","medium","high"] = "medium"

class GoalGene(BaseModel):
    explicitness: Literal["low","medium","high"] = "high"
    placement: Literal["top","after_context","bottom"] = "top"

class InstructionGene(BaseModel):
    style: Literal["imperative","declarative","conversational"] = "imperative"
    granularity: Literal["coarse","medium","fine"] = "medium"
    step_decomposition: bool = False

class ConstraintGene(BaseModel):
    placement: Literal["top","after_goal","bottom"] = "after_goal"
    explicitness: Literal["low","medium","high"] = "high"
    max_count: int = 8

class ExampleGene(BaseModel):
    enabled: bool = False
    count: int = 0
    selection: Literal["similar","diverse","hard_cases"] = "diverse"
    format: Literal["input_output","input_reasoning_output"] = "input_output"

class ReasoningGene(BaseModel):
    strategy: Literal["none","brief_plan","hidden_analysis","structured_checklist","decompose_then_answer"] = "none"
    visibility: Literal["visible","hidden"] = "hidden"
    budget: Literal["low","medium","high"] = "medium"

class VerificationGene(BaseModel):
    enabled: bool = False
    type: Literal["none","constraint_check","self_consistency","source_grounding_check","format_check"] = "none"
    position: Literal["before_final_output","after_final_output"] = "before_final_output"

class OutputGene(BaseModel):
    format: Literal["plain_text","markdown","json","json_schema","xml","table"] = "json_schema"
    strictness: Literal["low","medium","high"] = "high"
    include_schema_in_prompt: bool = True
    forbid_extra_fields: bool = True

class StyleGene(BaseModel):
    tone: Literal["professional","neutral","concise"] = "professional"
    verbosity: Literal["low","medium","high"] = "low"
    language: str = "zh"

class LayoutGene(BaseModel):
    section_order: list[str] = Field(default_factory=lambda: ["goal","constraints","input","reasoning_instruction","output_format"])
    delimiter: Literal["plain","markdown","xml","yaml"] = "xml"

class PromptGenome(BaseModel):
    genome_version: str = "1.0"
    task_id: str
    role: RoleGene = Field(default_factory=RoleGene)
    goal: GoalGene = Field(default_factory=GoalGene)
    instructions: InstructionGene = Field(default_factory=InstructionGene)
    constraints: ConstraintGene = Field(default_factory=ConstraintGene)
    examples: ExampleGene = Field(default_factory=ExampleGene)
    reasoning: ReasoningGene = Field(default_factory=ReasoningGene)
    verification: VerificationGene = Field(default_factory=VerificationGene)
    output: OutputGene = Field(default_factory=OutputGene)
    style: StyleGene = Field(default_factory=StyleGene)
    layout: LayoutGene = Field(default_factory=LayoutGene)
    search_space: dict[str, list[Any]] = Field(default_factory=dict)

    @classmethod
    def from_json(cls, path: str) -> "PromptGenome":
        import json
        with open(path, encoding="utf-8") as f:
            return cls.model_validate(json.load(f))
