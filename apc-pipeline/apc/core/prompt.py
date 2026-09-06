from __future__ import annotations
from typing import Any
from uuid import uuid4
from pydantic import BaseModel, Field


class CompiledPrompt(BaseModel):
    prompt_id: str = Field(default_factory=lambda: f"prompt_{uuid4().hex[:12]}")
    task_id: str
    genome_id: str
    model_id: str
    prompt_text: str
    template_version: str = "1.0"
    token_estimate: int | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
