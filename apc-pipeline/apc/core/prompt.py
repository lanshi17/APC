from __future__ import annotations
from typing import Any
from pydantic import BaseModel, Field
import uuid

class CompiledPrompt(BaseModel):
    prompt_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    genome_id: str
    model_id: str
    prompt_text: str
    template_version: str = "1.0"
    token_estimate: int | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
