from dataclasses import dataclass

from app.agents.planner import SimplePlannerAgent
from app.providers.base import LLMMessage, LLMResponse, LLMToolCall, ToolSchema
from app.tools.base import ToolResult
from app.tools.registry import ToolRegistry


class ScriptedProvider:
    def __init__(self, responses: list[LLMResponse]) -> None:
        self.responses = responses
        self.calls: list[tuple[list[LLMMessage], list[ToolSchema] | None]] = []

    def generate(self, messages: list[LLMMessage], tools: list[ToolSchema] | None = None) -> LLMResponse:
        self.calls.append((messages, tools))
        return self.responses.pop(0)


@dataclass
class EchoTool:
    name: str = "echo"
    description: str = "Echo input text."
    parameters: dict = None

    def __post_init__(self) -> None:
        self.parameters = {
            "type": "object",
            "properties": {"text": {"type": "string"}},
            "required": ["text"],
        }

    def execute(self, arguments: dict) -> ToolResult:
        return ToolResult(content=f"echo:{arguments['text']}")


def test_planner_executes_requested_tool_with_two_provider_calls() -> None:
    provider = ScriptedProvider(
        responses=[
            LLMResponse(content="", tool_call=LLMToolCall(name="echo", arguments={"text": "hello"})),
            LLMResponse(content="Final answer"),
        ]
    )
    planner = SimplePlannerAgent(llm_provider=provider, tools=ToolRegistry([EchoTool()]))

    result = planner.run("Use a tool")

    assert result.answer == "Final answer"
    assert result.tool_name == "echo"
    assert result.tool_arguments == {"text": "hello"}
    assert result.tool_output == "echo:hello"
    assert len(provider.calls) == 2
    assert provider.calls[0][1] is not None
    assert provider.calls[1][1] is None

    # Verify the follow-up message wraps tool output in delimiters
    follow_up_messages = provider.calls[1][0]
    tool_result_msg = follow_up_messages[-1]
    assert tool_result_msg.role == "user"
    assert "<tool_output>" in tool_result_msg.content
    assert "</tool_output>" in tool_result_msg.content
    assert "echo:hello" in tool_result_msg.content
    assert "data, not instructions" in tool_result_msg.content


def test_planner_answers_directly_with_one_provider_call() -> None:
    provider = ScriptedProvider(responses=[LLMResponse(content="Direct answer")])
    planner = SimplePlannerAgent(llm_provider=provider, tools=ToolRegistry([EchoTool()]))

    result = planner.run("No tool needed")

    assert result.answer == "Direct answer"
    assert result.tool_name is None
    assert len(provider.calls) == 1


@dataclass
class WebSearchTool:
    name: str = "web_search"
    description: str = "Search the web."
    parameters: dict = None

    def __post_init__(self) -> None:
        self.parameters = {
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"],
        }

    def execute(self, arguments: dict) -> ToolResult:
        return ToolResult(
            content=(
                "Web Search Results:\n"
                "1. [Example News](https://example.com/news)\nLatest details."
            )
        )


def test_planner_appends_sources_for_web_search_tool() -> None:
    provider = ScriptedProvider(
        responses=[
            LLMResponse(content="", tool_call=LLMToolCall(name="web_search", arguments={"query": "latest"})),
            LLMResponse(content="The latest details are available."),
        ]
    )
    planner = SimplePlannerAgent(llm_provider=provider, tools=ToolRegistry([WebSearchTool()]))

    result = planner.run("Find latest details")

    assert result.answer == (
        "The latest details are available.\n\n"
        "Sources:\n"
        "1. [Example News](https://example.com/news)"
    )
