from dataclasses import dataclass
from typing import Protocol

class LLMGenerationError(RuntimeError):
    """Raised when an LLM provider fails to produce a usable generation."""



@dataclass(slots=True)
class LLMImage:
    mime_type: str
    data: str


@dataclass(slots=True)
class LLMMessage:
    role: str
    content: str
    images: list[LLMImage] | None = None


@dataclass(slots=True)
class LLMResponse:
    content: str
    tool_call: "LLMToolCall | None" = None
    thought: str | None = None


@dataclass(slots=True)
class LLMToolCall:
    name: str
    arguments: dict


@dataclass(slots=True)
class ToolSchema:
    name: str
    description: str
    parameters: dict


class LLMProvider(Protocol):
    def generate(self, messages: list[LLMMessage], tools: list[ToolSchema] | None = None) -> LLMResponse:
        raise NotImplementedError
