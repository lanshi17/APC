from __future__ import annotations
import os, httpx
from tenacity import retry, stop_after_attempt, wait_exponential
from apc.models.base import BaseModelClient

class OpenAIClient(BaseModelClient):
    def __init__(self, model_id: str = "gpt-4o-mini", api_key: str | None = None, base_url: str | None = None):
        self._model_id = model_id
        self.api_key = api_key or os.getenv("OPENAI_API_KEY", "")
        self.base_url = (base_url or os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")).rstrip("/")
        self.client = httpx.Client(timeout=60)
    @property
    def model_id(self) -> str: return self._model_id
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=10))
    def complete(self, prompt: str, temperature: float = 0.0) -> str:
        resp = self.client.post(
            f"{self.base_url}/chat/completions",
            headers={"Authorization": f"Bearer {self.api_key}"},
            json={"model": self._model_id, "messages": [{"role":"user","content": prompt}], "temperature": temperature},
        )
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]
