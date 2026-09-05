import json
import logging
import time

import httpx
from langsmith import traceable

from app.core.config import settings
from app.providers.base import LLMGenerationError, LLMMessage, LLMProvider, LLMResponse, LLMToolCall, ToolSchema


logger = logging.getLogger(__name__)
RETRYABLE_STATUS_CODES = {429, 500, 502, 503, 504}


class GeminiProvider(LLMProvider):
    def __init__(
        self,
        api_key: str,
        model: str = "gemini-3.5-flash",
        max_output_tokens: int | None = None,
    ) -> None:
        self.api_key = api_key
        self.model = model
        self.max_output_tokens = max_output_tokens if max_output_tokens is not None else settings.llm_max_output_tokens

    @traceable(name="gemini_generate", run_type="llm")
    def generate(self, messages: list[LLMMessage], tools: list[ToolSchema] | None = None) -> LLMResponse:
        if not self.api_key:
            return LLMResponse(
                content=(
                    "LLM_API_KEY is not configured. The Phase 1 pipeline is working, "
                    "but set a Gemini key in apps/api/.env to get a real model response."
                )
            )

        url = (
            f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}:generateContent"
            f"?key={self.api_key}"
        )
        payload: dict = {
            "contents": self._contents(messages),
            "generationConfig": {
                "maxOutputTokens": self.max_output_tokens,
                "thinkingConfig": {"includeThoughts": False},
            },
        }
        if tools:
            payload["tools"] = [{"functionDeclarations": [self._function_declaration(tool) for tool in tools]}]

        response = self._post_generate_content(url=url, payload=payload)
        data = response.json()
        logger.debug("Gemini raw response: %s", data)
        candidates = data.get("candidates", [])
        if not candidates:
            return LLMResponse(content="The model returned no candidates.")
        candidate = candidates[0]
        finish_reason = candidate.get("finishReason")
        parts = candidate.get("content", {}).get("parts", [])

        thought_parts = [part.get("text", "") for part in parts if part.get("thought")]
        text_parts = [part.get("text", "") for part in parts if not part.get("thought")]
        
        raw_text = "".join(text_parts).strip()
        extracted_thought, clean_text = self._extract_thought_tags(raw_text)
        
        combined_thought = None
        if thought_parts or extracted_thought:
            combined_thought = "\n".join(filter(None, ["".join(thought_parts).strip(), extracted_thought])).strip()

        for part in parts:
            function_call = part.get("functionCall")
            if function_call:
                return LLMResponse(
                    content="",
                    tool_call=LLMToolCall(
                        name=function_call.get("name", ""),
                        arguments=function_call.get("args") or {},
                    ),
                    thought=combined_thought,
                )

        if finish_reason == "MAX_TOKENS" and not clean_text:
            raise LLMGenerationError(
                f"Gemini stopped with finishReason=MAX_TOKENS before producing an answer (model={self.model})."
            )
        return LLMResponse(content=clean_text or "The model returned an empty response.", thought=combined_thought)

    @staticmethod
    def _extract_thought_tags(raw_text: str) -> tuple[str | None, str]:
        import re
        thought_match = re.search(r"<thought>(.*?)</thought>", raw_text, flags=re.DOTALL | re.IGNORECASE)
        if thought_match:
            thought = thought_match.group(1).strip()
            clean_text = re.sub(r"<thought>.*?</thought>", "", raw_text, flags=re.DOTALL | re.IGNORECASE).strip()
            return thought, clean_text
        return None, raw_text.strip()

    def _post_generate_content(self, url: str, payload: dict) -> httpx.Response:
        max_attempts = 4
        for attempt in range(max_attempts):
            try:
                response = httpx.post(
                    url,
                    json=payload,
                    timeout=30,
                )
                response.raise_for_status()
                return response
            except httpx.HTTPStatusError as exc:
                status_code = exc.response.status_code
                reason = exc.response.reason_phrase
                if status_code in RETRYABLE_STATUS_CODES and attempt < max_attempts - 1:
                    sleep_time = 2 ** attempt
                    logger.warning("Gemini returned %d %s, retrying attempt %d/%d in %ds...", status_code, reason, attempt + 1, max_attempts, sleep_time)
                    time.sleep(sleep_time)
                    continue
                raise LLMGenerationError(
                    f"Gemini request failed with HTTP {status_code} {reason} (model={self.model})."
                ) from exc
            except httpx.HTTPError as exc:
                if attempt < max_attempts - 1:
                    sleep_time = 2 ** attempt
                    time.sleep(sleep_time)
                    continue
                raise LLMGenerationError(
                    f"Gemini request failed with {exc.__class__.__name__} (model={self.model})."
                ) from exc

        raise LLMGenerationError(f"Gemini request failed after {max_attempts} attempts (model={self.model}).")

    @staticmethod
    def _contents(messages: list[LLMMessage]) -> list[dict]:
        contents = []
        for message in messages:
            role = "model" if message.role == "assistant" else "user"
            parts = [{"text": f"{message.role}: {message.content}"}]
            for image in message.images or []:
                parts.append({
                    "inlineData": {
                        "mimeType": image.mime_type,
                        "data": image.data,
                    }
                })
            contents.append({"role": role, "parts": parts})
        return contents

    @staticmethod
    def _function_declaration(tool: ToolSchema) -> dict:
        parameters = json.loads(json.dumps(tool.parameters))
        parameters.pop("additionalProperties", None)
        return {
            "name": tool.name,
            "description": tool.description,
            "parameters": parameters,
        }
