from __future__ import annotations
from abc import ABC, abstractmethod

class BaseModelClient(ABC):
    @abstractmethod
    def complete(self, prompt: str, temperature: float = 0.0) -> str: ...
    @property
    @abstractmethod
    def model_id(self) -> str: ...
