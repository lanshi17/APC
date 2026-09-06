from __future__ import annotations
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4
from pydantic import BaseModel, Field


class TrialResult(BaseModel):
    trial_id: str = Field(default_factory=lambda: f"trial_{uuid4().hex[:12]}")
    task_id: str
    model_id: str
    model_version: str | None = None
    genome_id: str
    parent_genome_id: str | None = None
    mutation_note: str | None = None
    prompt_id: str
    dataset_id: str
    dataset_version: str | None = None
    score: float
    accuracy: float
    instruction_following: float
    format_score: float
    constraint_score: float
    robustness: float
    efficiency_score: float
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    avg_latency_ms: int = 0
    latency_ms: int = 0
    variance: float = 0.0
    format_error_rate: float = 0.0
    judge_id: str | None = None
    case_results: list[dict[str, Any]] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
