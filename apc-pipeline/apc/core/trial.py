from __future__ import annotations
from typing import Any
from pydantic import BaseModel, Field

class TrialResult(BaseModel):
    trial_id: str
    task_id: str
    model_id: str
    genome_id: str
    prompt_id: str
    dataset_id: str
    score: float
    accuracy: float
    instruction_following: float
    format_score: float
    constraint_score: float
    robustness: float
    efficiency_score: float
    total_input_tokens: int
    total_output_tokens: int
    latency_ms: int
    variance: float
    case_results: list[dict[str, Any]] = Field(default_factory=list)
