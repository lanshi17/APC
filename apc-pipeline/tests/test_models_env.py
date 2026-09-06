"""模型工厂环境变量层测试：yaml 覆盖 / 纯 env 定义新模型。"""
from __future__ import annotations

from apc.models.factory import (
    apply_env_overrides,
    create_client,
    has_credentials,
    load_model_config,
)
from apc.models.mock_client import MockClient
from apc.models.openai_client import OpenAIClient


def test_yaml_override_model_and_base_url(monkeypatch):
    monkeypatch.setenv("APC_GLM_MODEL", "glm-4-plus")
    monkeypatch.setenv("APC_GLM_BASE_URL", "https://gateway.example.com/v1")
    monkeypatch.setenv("APC_GLM_MAX_TOKENS", "4000")
    cfg = load_model_config("glm")
    assert cfg["model"] == "glm-4-plus"
    assert cfg["api_base"] == "https://gateway.example.com/v1"
    assert cfg["max_tokens"] == 4000
    # 未覆盖字段沿用 yaml
    assert cfg["provider"] == "zhipu"


def test_empty_override_keeps_yaml(monkeypatch):
    monkeypatch.setenv("APC_GLM_MODEL", "")
    assert load_model_config("glm")["model"] == "glm-4"


def test_custom_model_from_env_without_yaml(monkeypatch):
    monkeypatch.setenv("APC_DEEPSEEK_MODEL", "deepseek-chat")
    monkeypatch.setenv("APC_DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1")
    monkeypatch.setenv("APC_DEEPSEEK_API_KEY", "sk-test")
    cfg = load_model_config("deepseek")
    assert cfg["provider"] == "custom" and cfg["model"] == "deepseek-chat"
    assert has_credentials(cfg)
    client = create_client("deepseek")
    assert isinstance(client, OpenAIClient) and client.model == "deepseek-chat"


def test_custom_model_without_credentials_falls_back_to_mock(monkeypatch):
    monkeypatch.setenv("APC_DEEPSEEK_MODEL", "deepseek-chat")
    monkeypatch.setenv("APC_DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1")
    monkeypatch.delenv("APC_DEEPSEEK_API_KEY", raising=False)
    assert isinstance(create_client("deepseek"), MockClient)


def test_missing_model_still_mock():
    assert isinstance(create_client("nonexistent_model_xyz"), MockClient)


def test_bad_int_override_rejected(monkeypatch):
    import pytest

    monkeypatch.setenv("APC_GLM_MAX_TOKENS", "not-a-number")
    with pytest.raises(ValueError):
        apply_env_overrides("glm", {"model_id": "glm"})
