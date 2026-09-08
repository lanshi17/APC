from __future__ import annotations
import os
import time
import httpx
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from apc.models.base import BaseModelClient, CallResult

# 凭证解析不设供应商特例:模型配置(api_key / api_key_env)或全局 APC_API_KEY,三选一。


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


def resolve_api_key(model_config: dict) -> str:
    """凭证优先级: 直填 api_key > api_key_env 指向的环境变量。
    供应商名只是标签;换网关只需改 BASE_URL,换 key 变量改 api_key_env。"""
    direct = model_config.get("api_key") or ""
    if direct:
        return direct
    env_name = model_config.get("api_key_env") or ""
    return os.getenv(env_name, "") if env_name else ""


def build_openai_client(model_config: dict, api_key: str | None = None) -> OpenAIClient:
    """从模型配置 dict 构建客户端;凭证解析见 resolve_api_key。"""
    key = api_key or resolve_api_key(model_config)
    return OpenAIClient(
        model_id=model_config["model_id"], model=model_config["model"], base_url=model_config["api_base"],
        api_key=key, max_tokens=int(model_config.get("max_tokens", 2000)),
    )
