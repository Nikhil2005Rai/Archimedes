import pytest
import httpx

from app.core.config import settings
from app.providers.base import LLMGenerationError, LLMImage, LLMMessage
from app.providers.gemini import GeminiProvider


class FakeGeminiResponse:
    def __init__(self, data: dict | None = None) -> None:
        self.data = data or {
            "candidates": [
                {
                    "content": {
                        "parts": [
                            {"text": "I should decide whether to route.\n", "thought": True},
                            {"text": "ROUTE: knowledge"},
                        ]
                    }
                }
            ]
        }

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict:
        return self.data


def test_gemini_provider_filters_thought_parts_and_sends_generation_config(monkeypatch) -> None:
    captured_payload = {}

    def fake_post(url: str, json: dict, timeout: int):
        captured_payload["json"] = json
        return FakeGeminiResponse()

    monkeypatch.setattr("app.providers.gemini.httpx.post", fake_post)

    response = GeminiProvider(api_key="gemini-key", model="gemini-3.5-flash").generate(
        [LLMMessage(role="user", content="Use my uploaded knowledge")]
    )

    assert response.content == "ROUTE: knowledge"
    assert captured_payload["json"]["generationConfig"]["thinkingConfig"]["includeThoughts"] is False
    assert captured_payload["json"]["generationConfig"]["maxOutputTokens"] == settings.llm_max_output_tokens


def test_gemini_provider_uses_constructor_max_output_tokens(monkeypatch) -> None:
    captured_payload = {}

    def fake_post(url: str, json: dict, timeout: int):
        captured_payload["json"] = json
        return FakeGeminiResponse({"candidates": [{"content": {"parts": [{"text": "ok"}]}}]})

    monkeypatch.setattr("app.providers.gemini.httpx.post", fake_post)

    response = GeminiProvider(
        api_key="gemini-key",
        model="gemini-3.5-flash",
        max_output_tokens=512,
    ).generate([LLMMessage(role="user", content="Hello")])

    assert response.content == "ok"
    assert captured_payload["json"]["generationConfig"]["maxOutputTokens"] == 512


def test_gemini_provider_sends_inline_image_parts(monkeypatch) -> None:
    captured_payload = {}

    def fake_post(url: str, json: dict, timeout: int):
        captured_payload["json"] = json
        return FakeGeminiResponse({"candidates": [{"content": {"parts": [{"text": "image answer"}]}}]})

    monkeypatch.setattr("app.providers.gemini.httpx.post", fake_post)

    response = GeminiProvider(api_key="gemini-key", model="gemini-3.5-flash").generate(
        [
            LLMMessage(
                role="user",
                content="Describe this",
                images=[LLMImage(mime_type="image/png", data="aGVsbG8=")],
            )
        ]
    )

    assert response.content == "image answer"
    parts = captured_payload["json"]["contents"][0]["parts"]
    assert parts[0] == {"text": "user: Describe this"}
    assert parts[1] == {"inlineData": {"mimeType": "image/png", "data": "aGVsbG8="}}


def test_gemini_provider_raises_when_max_tokens_produces_no_answer(monkeypatch) -> None:
    def fake_post(url: str, json: dict, timeout: int):
        return FakeGeminiResponse(
            {
                "candidates": [
                    {
                        "finishReason": "MAX_TOKENS",
                        "content": {"parts": [{"text": "internal reasoning only", "thought": True}]},
                    }
                ]
            }
        )

    monkeypatch.setattr("app.providers.gemini.httpx.post", fake_post)

    with pytest.raises(LLMGenerationError, match="MAX_TOKENS"):
        GeminiProvider(api_key="gemini-key", model="gemini-3.5-flash").generate(
            [LLMMessage(role="user", content="Use my uploaded knowledge")]
        )


def test_gemini_provider_sanitizes_http_errors(monkeypatch) -> None:
    def fake_post(url: str, json: dict, timeout: int):
        request = httpx.Request("POST", url)
        response = httpx.Response(503, request=request)
        raise httpx.HTTPStatusError("upstream failed", request=request, response=response)

    monkeypatch.setattr("app.providers.gemini.httpx.post", fake_post)

    with pytest.raises(LLMGenerationError) as exc_info:
        GeminiProvider(api_key="secret-key", model="gemini-3.5-flash").generate(
            [LLMMessage(role="user", content="Hello")]
        )

    message = str(exc_info.value)
    assert "HTTP 503 Service Unavailable" in message
    assert "secret-key" not in message
    assert "?key=" not in message


def test_build_provider_unsupported_error_message() -> None:
    from app.providers.registry import build_provider
    with pytest.raises(ValueError) as exc_info:
        build_provider(api_key="key", provider_name="invalid_provider")
    
    assert "Unsupported LLM_PROVIDER='invalid_provider'" in str(exc_info.value)
