"""Shared LLM plumbing for the P3 extended corpus (Scenario A / B / C).

Everything the extended corpus contains is produced by *real* model calls
against the DashScope OpenAI-compatible endpoint, then cached on disk so a
re-run is free and byte-reproducible.

The P3 plan asked for ``gpt-4o`` / ``gpt-4o-vision``.  No OpenAI credential is
configured in this environment (``OPENAI_API_KEY`` is empty); the only usable
credential is ``DASHSCOPE_API_KEY``, whose key restrictions allow exactly two
of the 262 catalogued chat models: ``qwen3.8-flash`` (primary) and
``qwen3.8-27b`` (secondary / contrast).  Both accept image inputs.  The
substitution is recorded in every artifact under ``provenance.models`` and in
``docs/active/P3_EXTENDED_CORPUS.md``.

Caching
-------
Cache key = sha256 of (model, messages, temperature, max_tokens, seed, tag).
A cache hit returns the *stored* server response, so generated text, judge
scores and token accounting are all reproducible from the on-disk cache alone.
"""
from __future__ import annotations

import base64
import hashlib
import json
import mimetypes
import os
import re
import threading
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

import httpx

from .data import APC_ROOT

# ---------------------------------------------------------------------------
# endpoints / models
# ---------------------------------------------------------------------------
DEFAULT_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
PRIMARY_MODEL = "qwen3.8-flash"      # gpt-4o substitute (text + vision)
SECONDARY_MODEL = "qwen3.8-27b"      # gpt-4o substitute, contrast model
EMBED_MODEL = "qwen3.7-text-embedding"

CORPUS_DIR = Path(__file__).resolve().parent / "corpus_extended"
CACHE_DIR = CORPUS_DIR / "_cache" / "llm"
LEDGER = CORPUS_DIR / "_cache" / "usage.jsonl"

_RETRY_STATUS = {408, 409, 429, 500, 502, 503, 504, 524}


# ---------------------------------------------------------------------------
# credentials
# ---------------------------------------------------------------------------
def load_env() -> Dict[str, str]:
    """Parse ``<code-repo>/.env`` (the same file the rest of the repo uses)."""
    env: Dict[str, str] = {}
    dotenv = APC_ROOT / ".env"
    if dotenv.exists():
        for line in dotenv.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                env[k.strip()] = v.strip()
    merged = dict(env)
    for k, v in os.environ.items():          # real environment wins over .env
        if k in merged or k.startswith(("APC_", "DASHSCOPE", "OPENAI", "JEV_")):
            merged[k] = v
    return merged


def api_key(env: Optional[Dict[str, str]] = None) -> str:
    env = env or load_env()
    key = env.get("DASHSCOPE_API_KEY", "") or os.environ.get("DASHSCOPE_API_KEY", "")
    if not key:
        raise RuntimeError("DASHSCOPE_API_KEY missing (checked .env and environment)")
    return key


# ---------------------------------------------------------------------------
# client
# ---------------------------------------------------------------------------
def data_uri(path: Path) -> str:
    """Encode a local image as a data URI (DashScope accepts these inline)."""
    mime = mimetypes.guess_type(path.name)[0] or "image/jpeg"
    return f"data:{mime};base64,{base64.b64encode(path.read_bytes()).decode()}"


def image_part(image: "str | Path") -> Dict[str, Any]:
    """Build one ``image_url`` content part from a URL or a local file."""
    if isinstance(image, Path) or (isinstance(image, str) and Path(image).exists()):
        url = data_uri(Path(image))
    else:
        url = str(image)
    return {"type": "image_url", "image_url": {"url": url}}


def text_part(text: str) -> Dict[str, Any]:
    return {"type": "text", "text": text}


class LLMError(RuntimeError):
    pass


class LLMClient:
    """Thin OpenAI-compatible client with disk cache, retries and a usage ledger."""

    def __init__(self, base_url: str = DEFAULT_BASE_URL, key: Optional[str] = None,
                 cache_dir: Path = CACHE_DIR, timeout: float = 240.0,
                 max_retries: int = 5, use_cache: bool = True):
        self.base_url = base_url.rstrip("/")
        self.key = key or api_key()
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.timeout = timeout
        self.max_retries = max_retries
        self.use_cache = use_cache
        self._lock = threading.Lock()
        self.n_calls = 0
        self.n_cache_hits = 0
        self.tokens_in = 0
        self.tokens_out = 0

    # -- internals --------------------------------------------------------
    @staticmethod
    def _key(payload: dict) -> str:
        blob = json.dumps(payload, sort_keys=True, ensure_ascii=False, default=str)
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:24]

    def _cache_path(self, key: str) -> Path:
        return self.cache_dir / f"{key}.json"

    def _ledger(self, row: dict) -> None:
        LEDGER.parent.mkdir(parents=True, exist_ok=True)
        with self._lock:
            with LEDGER.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")

    # -- chat -------------------------------------------------------------
    def chat(self, messages: Sequence[Dict[str, Any]], model: str = PRIMARY_MODEL,
             temperature: float = 0.0, max_tokens: int = 1024,
             seed: Optional[int] = None, tag: str = "",
             response_format: Optional[dict] = None,
             extra: Optional[dict] = None) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "model": model, "messages": list(messages),
            "temperature": temperature, "max_tokens": max_tokens,
        }
        if seed is not None:
            payload["seed"] = seed
        if response_format is not None:
            payload["response_format"] = response_format
        # `extra` carries non-OpenAI knobs (e.g. DashScope's enable_thinking) and
        # is part of the cache key so thinking / non-thinking runs never collide.
        if extra:
            payload.update(extra)
        key = self._key({**payload, "tag": tag})
        path = self._cache_path(key)

        if self.use_cache and path.exists():
            self.n_cache_hits += 1
            rec = json.loads(path.read_text(encoding="utf-8"))
            rec["cached"] = True
            return rec

        body = dict(payload)
        last_err: Optional[Exception] = None
        t0 = time.time()
        for attempt in range(self.max_retries):
            try:
                r = httpx.post(f"{self.base_url}/chat/completions",
                               headers={"Authorization": f"Bearer {self.key}",
                                        "Content-Type": "application/json"},
                               json=body, timeout=self.timeout)
                if r.status_code in _RETRY_STATUS:
                    raise LLMError(f"HTTP {r.status_code}: {r.text[:200]}")
                if r.status_code != 200:
                    raise LLMError(f"HTTP {r.status_code}: {r.text[:400]}")
                data = r.json()
                choice = (data.get("choices") or [{}])[0]
                msg = choice.get("message") or {}
                usage = data.get("usage") or {}
                rec = {
                    "tag": tag,
                    "model_requested": model,
                    "model_returned": data.get("model"),
                    "text": msg.get("content") or "",
                    "reasoning": msg.get("reasoning_content") or "",
                    "finish_reason": choice.get("finish_reason"),
                    "usage": {"prompt_tokens": int(usage.get("prompt_tokens", 0)),
                              "completion_tokens": int(usage.get("completion_tokens", 0))},
                    "response_id": data.get("id"),
                    "latency_s": round(time.time() - t0, 3),
                    "cached": False,
                    "cache_key": key,
                }
                if self.use_cache:
                    path.write_text(json.dumps(rec, ensure_ascii=False, indent=1),
                                    encoding="utf-8")
                self.n_calls += 1
                self.tokens_in += rec["usage"]["prompt_tokens"]
                self.tokens_out += rec["usage"]["completion_tokens"]
                self._ledger({"ts": round(time.time(), 1), "tag": tag, "model": model,
                              "cache_key": key, **rec["usage"]})
                return rec
            except Exception as e:  # noqa: BLE001
                last_err = e
                time.sleep(min(2.0 * (attempt + 1), 20.0))
        raise LLMError(f"chat failed after {self.max_retries} tries: {last_err}")

    def chat_text(self, prompt: str, **kw) -> str:
        return self.chat([{"role": "user", "content": prompt}], **kw)["text"]

    # -- embeddings -------------------------------------------------------
    def embed(self, texts: Sequence[str], model: str = EMBED_MODEL,
              tag: str = "") -> List[List[float]]:
        payload = {"model": model, "input": list(texts)}
        key = self._key({**payload, "tag": tag, "kind": "embed"})
        path = self._cache_path(key)
        if self.use_cache and path.exists():
            self.n_cache_hits += 1
            return json.loads(path.read_text(encoding="utf-8"))["embeddings"]
        last_err: Optional[Exception] = None
        for attempt in range(self.max_retries):
            try:
                r = httpx.post(f"{self.base_url}/embeddings",
                               headers={"Authorization": f"Bearer {self.key}",
                                        "Content-Type": "application/json"},
                               json=payload, timeout=self.timeout)
                if r.status_code != 200:
                    raise LLMError(f"HTTP {r.status_code}: {r.text[:300]}")
                data = r.json()
                vecs = [d["embedding"] for d in data["data"]]
                if self.use_cache:
                    path.write_text(json.dumps({"embeddings": vecs}), encoding="utf-8")
                self.n_calls += 1
                usage = data.get("usage") or {}
                self.tokens_in += int(usage.get("prompt_tokens", 0) or usage.get("total_tokens", 0) or 0)
                self._ledger({"ts": round(time.time(), 1), "tag": tag or "embed",
                              "model": model, "cache_key": key, "kind": "embed",
                              **{k: int(v) for k, v in usage.items() if isinstance(v, int)}})
                return vecs
            except Exception as e:  # noqa: BLE001
                last_err = e
                time.sleep(min(2.0 * (attempt + 1), 20.0))
        raise LLMError(f"embed failed after {self.max_retries} tries: {last_err}")

    # -- json helper ------------------------------------------------------
    def chat_json(self, prompt: str, model: str = PRIMARY_MODEL,
                  temperature: float = 0.0, max_tokens: int = 512,
                  seed: Optional[int] = None, tag: str = "",
                  extra: Optional[dict] = None) -> Any:
        """Ask for JSON and parse it tolerantly (models occasionally wrap it)."""
        return self.chat_json_content([{"type": "text", "text": prompt}],
                                      model=model, temperature=temperature,
                                      max_tokens=max_tokens, seed=seed, tag=tag,
                                      extra=extra)

    def chat_json_content(self, content: "str | list", model: str = PRIMARY_MODEL,
                          temperature: float = 0.0, max_tokens: int = 512,
                          seed: Optional[int] = None, tag: str = "",
                          extra: Optional[dict] = None) -> Any:
        """JSON-mode call with either a plain string or a multimodal content list."""
        if isinstance(content, str):
            content = [{"type": "text", "text": content}]
        rec = self.chat([{"role": "user", "content": content}], model=model,
                        temperature=temperature, max_tokens=max_tokens, seed=seed,
                        tag=tag, response_format={"type": "json_object"}, extra=extra)
        return extract_json(rec["text"]), rec

    def stats(self) -> Dict[str, Any]:
        return {"calls": self.n_calls, "cache_hits": self.n_cache_hits,
                "prompt_tokens": self.tokens_in, "completion_tokens": self.tokens_out}


def extract_score(obj: Any, keys: Sequence[str] = ("score", "rating", "value", "points")) -> Optional[float]:
    """Pull a numeric score out of whatever shape the judge replied with.

    Observed shapes: ``{"score": 7}``, ``7``, ``[{"score": 7}]`` (JSON-mode
    sometimes wraps a single object in a list).
    """
    if isinstance(obj, list):
        for item in obj:
            got = extract_score(item, keys)
            if got is not None:
                return got
        return None
    if isinstance(obj, dict):
        for k in keys:
            if k in obj:
                try:
                    return float(obj[k])
                except (TypeError, ValueError):
                    pass
        if len(obj) == 1:
            try:
                return float(next(iter(obj.values())))
            except (TypeError, ValueError):
                return None
        return None
    if isinstance(obj, bool):
        return None
    if isinstance(obj, (int, float)):
        return float(obj)
    if isinstance(obj, str):
        m = re.search(r"-?\d+(?:\.\d+)?", obj)
        if m:
            return float(m.group(0))
    return None


def extract_json(text: str) -> Any:
    text = (text or "").strip()
    if text.startswith("```"):
        text = text.split("```")[1]
        if text.lstrip().lower().startswith("json"):
            text = text.lstrip()[4:]
        text = text.strip()
    try:
        return json.loads(text)
    except Exception:
        pass
    for opener, closer in (("{", "}"), ("[", "]")):
        i, j = text.find(opener), text.rfind(closer)
        if i >= 0 and j > i:
            try:
                return json.loads(text[i:j + 1])
            except Exception:
                continue
    raise LLMError(f"could not parse JSON from: {text[:200]!r}")


def cosine(a: Sequence[float], b: Sequence[float]) -> float:
    import math
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else 0.0


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
