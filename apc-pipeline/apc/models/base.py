from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any
from pydantic import BaseModel, Field


class CallResult(BaseModel):
    """一次模型调用的完整记录（FR-3）。"""
    text: str
    model_id: str
    model_version: str | None = None
    temperature: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0
    latency_ms: int = 0
    retries: int = 0
    finish_reason: str | None = None
    raw: dict[str, Any] | None = Field(default=None, exclude=True)

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens


class BaseModelClient(ABC):
    @abstractmethod
    def complete(self, prompt: str, temperature: float = 0.0) -> CallResult: ...

    @property
    @abstractmethod
    def model_id(self) -> str: ...

    @property
    def model_version(self) -> str | None:
        return None
