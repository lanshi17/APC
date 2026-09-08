from __future__ import annotations

import os
import re
from pathlib import Path

import yaml

from apc.models.base import BaseModelClient
from apc.models.mock_client import MockClient
from apc.models.openai_client import OpenAIClient, build_openai_client

_REPO_ROOT = Path(__file__).resolve().parents[3]

# 仓库 .env 是唯一凭证来源:override=True 防止 shell 里残留的同名旧值遮蔽 .env;
# 所有入口(CLI / python -m / scripts)只需 import factory 即自动加载。
from dotenv import load_dotenv  # noqa: E402

load_dotenv(_REPO_ROOT / ".env", override=True)

# 环境变量后缀 → 配置字段；APC_<MODEL_ID 大写>_<后缀> 优先级高于 yaml。
_ENV_FIELDS = {
    "PROVIDER": "provider",
    "BASE_URL": "api_base",
    "API_BASE": "api_base",
    "MODEL": "model",
    "NAME": "name",
    "API_KEY": "api_key",
    "API_KEY_ENV": "api_key_env",
    "MAX_TOKENS": "max_tokens",
    "TEMPERATURE": "temperature",
}


def _env_prefix(model_id: str) -> str:
    return "APC_" + re.sub(r"[^A-Za-z0-9]+", "_", model_id).upper().strip("_") + "_"


def apply_env_overrides(model_id: str, cfg: dict) -> dict:
    """环境变量覆盖 yaml（只覆盖非空值；MAX_TOKENS/TEMPERATURE 做类型转换）。"""
    prefix = _env_prefix(model_id)
    out = dict(cfg)
    for suffix, key in _ENV_FIELDS.items():
        val = os.getenv(prefix + suffix)
        if val is None or val == "":
            continue
        if key == "max_tokens":
            try:
                out[key] = int(val)
            except ValueError:
                raise ValueError(f"{prefix + suffix} 必须是整数，当前为 {val!r}")
        elif key == "temperature":
            try:
                out[key] = float(val)
            except ValueError:
                raise ValueError(f"{prefix + suffix} 必须是数值，当前为 {val!r}")
        else:
            out[key] = val
    return out


def config_from_env(model_id: str) -> dict | None:
    """无 yaml 时凭环境变量定义全新模型；MODEL + BASE_URL 必填，否则返回 None。"""
    prefix = _env_prefix(model_id)
    model = os.getenv(prefix + "MODEL")
    base = os.getenv(prefix + "BASE_URL") or os.getenv(prefix + "API_BASE")
    if not model or not base:
        return None
    try:
        temperature = float(os.getenv(prefix + "TEMPERATURE", "0.0"))
        max_tokens = int(os.getenv(prefix + "MAX_TOKENS", "2000"))
    except ValueError as e:
        raise ValueError(f"{prefix}TEMPERATURE/MAX_TOKENS 类型错误：{e}")
    return {
        "model_id": model_id,
        "provider": os.getenv(prefix + "PROVIDER", "custom"),
        "name": os.getenv(prefix + "NAME", model),
        "api_base": base,
        "model": model,
        "api_key_env": os.getenv(prefix + "API_KEY_ENV", ""),
        "api_key": os.getenv(prefix + "API_KEY", ""),
        "temperature": temperature,
        "max_tokens": max_tokens,
    }



def load_model_config(model_id: str, configs_dir: str | Path | None = None) -> dict:
    """yaml 优先；无 yaml 时回退纯环境变量定义（见 config_from_env）；
    最后统一叠加 APC_<ID>_* 覆盖层。"""
    base = Path(configs_dir) if configs_dir else _REPO_ROOT / "configs" / "models"
    path = base / f"{model_id}.yaml"
    if path.exists():
        with open(path, encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        if cfg.get("model_id") != model_id:
            raise ValueError(f"{path}: model_id 与文件名不一致")
    else:
        cfg = config_from_env(model_id)
        if cfg is None:
            raise FileNotFoundError(
                f"模型配置不存在: {path}（且未检测到 {model_id} 的 "
                f"{_env_prefix(model_id)}MODEL/{_env_prefix(model_id)}BASE_URL 环境变量）")
    return apply_env_overrides(model_id, cfg)


def has_credentials(cfg: dict) -> bool:
    from apc.models.openai_client import resolve_api_key
    return bool(resolve_api_key(cfg))


def create_client(model_id: str, configs_dir: str | Path | None = None,
                  prefer_mock: bool = False) -> BaseModelClient:
    """有凭证走 OpenAI-compatible 通路；无凭证回退 MockClient（可重复验证，FR-3）。"""
    if model_id.startswith("mock"):
        return MockClient(model_id)
    try:
        cfg = load_model_config(model_id, configs_dir)
    except FileNotFoundError:
        return MockClient(model_id)
    if prefer_mock or not has_credentials(cfg):
        return MockClient(model_id)
    client: OpenAIClient = build_openai_client(cfg)
    return client
