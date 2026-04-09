"""Tests for safe Ollama client behavior and JSON parsing."""

import pytest

from retail_forecasting.ollama_client import (
    OllamaClient,
    OllamaResponseError,
    OllamaUnavailableError,
    extract_response_text,
    parse_json_from_model_text,
)


def test_parse_json_from_plain_text_object() -> None:
    """Parser should accept direct JSON object content."""
    parsed = parse_json_from_model_text('{"plan_version": "1.0", "specs": []}')
    assert parsed["plan_version"] == "1.0"
    assert parsed["specs"] == []


def test_parse_json_from_fenced_block() -> None:
    """Parser should extract JSON from fenced markdown responses."""
    raw = """```json
    {"specs": [{"operation": "lag_feature", "feature_name": "units_lag_llm", "source_column": "units_sold", "lag": 1, "group_by": ["store_id", "product_id"]}]}
    ```"""
    parsed = parse_json_from_model_text(raw)

    assert isinstance(parsed, dict)
    assert isinstance(parsed.get("specs"), list)


def test_parse_json_from_list_normalizes_to_specs_object() -> None:
    """List responses should normalize into object form for downstream code."""
    parsed = parse_json_from_model_text('[{"operation": "lag_feature", "feature_name": "x1_llm", "source_column": "units_sold", "lag": 1, "group_by": ["store_id", "product_id"]}]')

    assert "specs" in parsed
    assert isinstance(parsed["specs"], list)
    assert len(parsed["specs"]) == 1


def test_malformed_json_response_is_handled_gracefully(monkeypatch) -> None:
    """Client should report malformed JSON responses without crashing."""
    client = OllamaClient(host="http://localhost:11434", model="test-model")

    def fake_chat(system_prompt: str, user_prompt: str) -> dict[str, object]:
        return {"message": {"content": "not-json-at-all"}}

    monkeypatch.setattr(client, "_chat", fake_chat)

    response = client.plan_feature_specs(system_prompt="sys", user_prompt="user")

    assert response.reachable is True
    assert response.parsed_json is None
    assert response.error is not None
    assert "not valid JSON" in response.error


def test_ollama_unavailable_returns_non_crashing_result(monkeypatch) -> None:
    """Transport errors should produce reachable=False instead of raising to caller."""
    client = OllamaClient(host="http://localhost:11434", model="test-model")

    def fake_chat(system_prompt: str, user_prompt: str) -> dict[str, object]:
        raise OllamaUnavailableError("connection refused")

    monkeypatch.setattr(client, "_chat", fake_chat)

    response = client.plan_feature_specs(system_prompt="sys", user_prompt="user")

    assert response.reachable is False
    assert response.parsed_json is None
    assert response.error is not None
    assert "connection refused" in response.error


def test_extract_response_text_requires_content() -> None:
    """Missing content fields should raise a response parsing error."""
    with pytest.raises(OllamaResponseError, match="did not include assistant content"):
        extract_response_text({"unexpected": "payload"})
