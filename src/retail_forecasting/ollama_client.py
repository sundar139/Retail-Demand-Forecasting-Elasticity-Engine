"""Safe Ollama client wrapper for structured JSON feature planning."""

from dataclasses import dataclass
import json
import logging
from urllib import error as urllib_error
from urllib import request as urllib_request

from retail_forecasting.config import get_settings

LOGGER = logging.getLogger(__name__)


class OllamaClientError(RuntimeError):
    """Base exception type for Ollama client failures."""


class OllamaUnavailableError(OllamaClientError):
    """Raised when Ollama host is unreachable."""


class OllamaResponseError(OllamaClientError):
    """Raised when Ollama returns malformed or incomplete data."""


@dataclass(frozen=True, slots=True)
class OllamaPlannerResponse:
    """Result object for planner requests against Ollama."""

    reachable: bool
    host: str
    model: str
    raw_response_text: str
    parsed_json: dict[str, object] | None
    error: str | None = None
    planner_model_available: bool | None = None


class OllamaClient:
    """Minimal Ollama JSON planning client using explicit request/parse flow."""

    def __init__(
        self,
        host: str | None = None,
        model: str | None = None,
        timeout_seconds: float = 30.0,
    ) -> None:
        settings = get_settings()
        self.host = (host or settings.ollama_host).rstrip("/")
        self.model = model or settings.ollama_model
        self.timeout_seconds = timeout_seconds

    def plan_feature_specs(self, system_prompt: str, user_prompt: str) -> OllamaPlannerResponse:
        """Request a structured JSON plan from Ollama and parse it safely."""
        planner_model_available = self.check_model_available()

        try:
            payload = self._chat(system_prompt=system_prompt, user_prompt=user_prompt)
        except OllamaUnavailableError as exc:
            return OllamaPlannerResponse(
                reachable=False,
                host=self.host,
                model=self.model,
                raw_response_text="",
                parsed_json=None,
                error=str(exc),
                planner_model_available=planner_model_available,
            )
        except OllamaResponseError as exc:
            return OllamaPlannerResponse(
                reachable=True,
                host=self.host,
                model=self.model,
                raw_response_text="",
                parsed_json=None,
                error=str(exc),
                planner_model_available=planner_model_available,
            )

        raw_text = extract_response_text(payload)
        try:
            parsed = parse_json_from_model_text(raw_text)
        except ValueError as exc:
            return OllamaPlannerResponse(
                reachable=True,
                host=self.host,
                model=self.model,
                raw_response_text=raw_text,
                parsed_json=None,
                error=(
                    "Ollama returned a response that is not valid JSON for feature specs. "
                    f"Reason: {exc}"
                ),
                planner_model_available=planner_model_available,
            )

        return OllamaPlannerResponse(
            reachable=True,
            host=self.host,
            model=self.model,
            raw_response_text=raw_text,
            parsed_json=parsed,
            error=None,
            planner_model_available=planner_model_available,
        )

    def check_model_available(self) -> bool | None:
        """Return whether the configured model appears in Ollama tags, or None if unknown."""
        endpoint = f"{self.host}/api/tags"
        request = urllib_request.Request(endpoint, method="GET")

        try:
            with urllib_request.urlopen(request, timeout=self.timeout_seconds) as response:
                body_text = response.read().decode("utf-8")
        except (urllib_error.HTTPError, urllib_error.URLError, TimeoutError):
            return None

        try:
            payload = json.loads(body_text)
        except json.JSONDecodeError:
            return None

        if not isinstance(payload, dict):
            return None

        models = payload.get("models")
        if not isinstance(models, list):
            return None

        configured = self.model.strip().lower()
        if not configured:
            return None

        for model_item in models:
            if not isinstance(model_item, dict):
                continue
            name = model_item.get("name")
            model_id = model_item.get("model")
            for candidate in (name, model_id):
                if isinstance(candidate, str) and candidate.strip().lower() == configured:
                    return True

        return False

    def _chat(self, system_prompt: str, user_prompt: str) -> dict[str, object]:
        """Call the Ollama chat endpoint and return parsed response JSON."""
        endpoint = f"{self.host}/api/chat"
        request_payload = {
            "model": self.model,
            "stream": False,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "options": {
                "temperature": 0,
            },
        }

        request_body = json.dumps(request_payload).encode("utf-8")
        request = urllib_request.Request(
            endpoint,
            data=request_body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        try:
            with urllib_request.urlopen(request, timeout=self.timeout_seconds) as response:
                body_text = response.read().decode("utf-8")
        except urllib_error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace") if hasattr(exc, "read") else ""
            raise OllamaResponseError(
                f"Ollama returned HTTP {exc.code}. Verify model availability ({self.model}) and endpoint "
                f"({endpoint}). Details: {detail[:400]}"
            ) from exc
        except urllib_error.URLError as exc:
            raise OllamaUnavailableError(
                f"Cannot reach Ollama host '{self.host}'. Ensure Ollama is running and OLLAMA_HOST is correct. "
                f"Network error: {exc.reason}"
            ) from exc
        except TimeoutError as exc:
            raise OllamaUnavailableError(
                f"Timed out calling Ollama at {endpoint}. Try increasing timeout or checking server load."
            ) from exc

        try:
            payload = json.loads(body_text)
        except json.JSONDecodeError as exc:
            raise OllamaResponseError(
                "Ollama HTTP response was not valid JSON. "
                f"First 200 chars: {body_text[:200]}"
            ) from exc

        if not isinstance(payload, dict):
            raise OllamaResponseError("Ollama response payload must be a JSON object")

        return payload


def extract_response_text(response_payload: dict[str, object]) -> str:
    """Extract model text content from either chat or generate endpoint payloads."""
    message = response_payload.get("message")
    if isinstance(message, dict):
        content = message.get("content")
        if isinstance(content, str) and content.strip():
            return content.strip()

    response_text = response_payload.get("response")
    if isinstance(response_text, str) and response_text.strip():
        return response_text.strip()

    raise OllamaResponseError("Ollama response did not include assistant content")


def parse_json_from_model_text(model_text: str) -> dict[str, object]:
    """Parse JSON object from model output with strict object-or-list normalization."""
    candidate_texts = _candidate_json_texts(model_text)
    last_error: Exception | None = None

    for candidate in candidate_texts:
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError as exc:
            last_error = exc
            continue

        if isinstance(parsed, dict):
            return parsed
        if isinstance(parsed, list):
            return {"specs": parsed}

        raise ValueError("Planner JSON must be an object or list of specs")

    if last_error is not None:
        raise ValueError(str(last_error)) from last_error

    raise ValueError("No valid JSON content found")


def _candidate_json_texts(model_text: str) -> list[str]:
    """Build JSON parse candidates from raw model text and fenced blocks."""
    cleaned = model_text.strip()
    candidates: list[str] = [cleaned]

    fence_start = cleaned.find("```")
    if fence_start >= 0:
        fence_end = cleaned.rfind("```")
        if fence_end > fence_start:
            fenced = cleaned[fence_start + 3 : fence_end].strip()
            if fenced.lower().startswith("json"):
                fenced = fenced[4:].strip()
            candidates.append(fenced)

    object_start = cleaned.find("{")
    object_end = cleaned.rfind("}")
    if object_start >= 0 and object_end > object_start:
        candidates.append(cleaned[object_start : object_end + 1])

    array_start = cleaned.find("[")
    array_end = cleaned.rfind("]")
    if array_start >= 0 and array_end > array_start:
        candidates.append(cleaned[array_start : array_end + 1])

    deduplicated: list[str] = []
    seen: set[str] = set()
    for item in candidates:
        if not item or item in seen:
            continue
        deduplicated.append(item)
        seen.add(item)

    return deduplicated
