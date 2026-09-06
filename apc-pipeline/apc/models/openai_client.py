from __future__ import annotations
import os
import time
import httpx
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from apc.models.base import BaseModelClient, CallResult

_PROVIDER_ENV_KEYS = {"openai": "OPENAI_API_KEY", "dashscope": "DASHSCOPE_API_KEY", "zhipu": "ZHIPUAI_API_KEY"}


class OpenAIClient(BaseModelClient):
    """OpenAI-compatible 调用通路（GPT / DashScope compatible-mode / Zhipu compatible API）。"""

    def __init__(self, model_id: str, model: str, base_url: str, api_key: str | None = None,
                 max_tokens: int = 2000, timeout: float = 60.0):
        self._model_id = model_id
        self.model = model
        self.max_tokens = max_tokens
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key or ""
        self.client = httpx.Client(timeout=timeout)
        self.last_usage: dict | None = None

    @property
    def model_id(self) -> str:
        return self._model_id

    @property
    def model_version(self) -> str | None:
        return self._version

    _version: str | None = None

    def _post(self, prompt: str, temperature: float) -> dict:
        resp = self.client.post(
            f"{self.base_url}/chat/completions",
            headers={"Authorization": f"Bearer {self.api_key}"} if self.api_key else {},
            json={"model": self.model, "messages": [{"role": "user", "content": prompt}],
                  "temperature": temperature, "max_tokens": self.max_tokens},
        )
        resp.raise_for_status()
        return resp.json()

    def complete(self, prompt: str, temperature: float = 0.0) -> CallResult:
        attempts = {"n": 0}
        start = time.monotonic()

        def _before(retry_state):
            attempts["n"] += 1

        @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=10),
               retry=retry_if_exception_type((httpx.HTTPError, KeyError)), before_sleep=_before)
        def _call() -> dict:
            return self._post(prompt, temperature)

        data = _call()
        latency_ms = int((time.monotonic() - start) * 1000)
        choice = data["choices"][0]
        usage = data.get("usage") or {}
        self._version = data.get("model")
        self.last_usage = usage
        return CallResult(
            text=choice["message"]["content"] or "",
            model_id=self._model_id, model_version=self._version, temperature=temperature,
            input_tokens=int(usage.get("prompt_tokens", 0)), output_tokens=int(usage.get("completion_tokens", 0)),
            latency_ms=latency_ms, retries=attempts["n"], finish_reason=choice.get("finish_reason"),
            raw={"response_id": data.get("id")},
        )


def build_openai_client(model_config: dict, api_key: str | None = None) -> OpenAIClient:
    """从 configs/models/*.yaml 的 dict 构建客户端；凭证只从环境变量读取。"""
    env_name = model_config.get("api_key_env") or _PROVIDER_ENV_KEYS.get(model_config.get("provider", ""), "OPENAI_API_KEY")
    key = api_key if api_key is not None else os.getenv(env_name, "")
    return OpenAIClient(
        model_id=model_config["model_id"], model=model_config["model"], base_url=model_config["api_base"],
        api_key=key, max_tokens=int(model_config.get("max_tokens", 2000)),
    )
