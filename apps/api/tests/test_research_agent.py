from dataclasses import dataclass

from app.agents.research import ResearchAgent
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
class FakeWebSearchTool:
    name: str = "web_search"
    description: str = "Search the web."
    parameters: dict = None

    def __post_init__(self) -> None:
        self.parameters = {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "max_results": {"type": "integer"},
            },
            "required": ["query"],
        }
        self.calls: list[dict] = []

    def execute(self, arguments: dict) -> ToolResult:
        self.calls.append(arguments)
        return ToolResult(
            content=(
                "Summary Answer: The live answer is available.\n\n"
                "Web Search Results:\n"
                "1. [Live Source](https://example.com/live)\nLatest information."
            )
        )


@dataclass
class FakeCurrentTimeTool:
    name: str = "current_time"
    description: str = "Get current time."
    parameters: dict = None

    def __post_init__(self) -> None:
        self.parameters = {"type": "object", "properties": {}}

    def execute(self, arguments: dict) -> ToolResult:
        return ToolResult(content="2026-09-05 10:00")


def test_research_agent_forces_web_search_when_model_does_not_call_tool() -> None:
    provider = ScriptedProvider(
        [
            LLMResponse(content="I cannot browse from here."),
            LLMResponse(content="Here is the latest answer."),
        ]
    )
    web_search = FakeWebSearchTool()
    agent = ResearchAgent(provider, ToolRegistry([web_search]))

    result = agent.run("Search the web for latest AI news")

    assert web_search.calls == [{"query": "Search the web for latest AI news", "max_results": 5}]
    assert result.tool_name == "web_search"
    assert result.tool_arguments == {"query": "Search the web for latest AI news", "max_results": 5}
    assert "Here is the latest answer." in result.answer
    assert "Sources:" in result.answer
    assert "[Live Source](https://example.com/live)" in result.answer
    assert len(provider.calls) == 2
    assert provider.calls[1][1] is None


def test_research_agent_forces_web_search_after_current_time_if_model_still_does_not_search() -> None:
    provider = ScriptedProvider(
        [
            LLMResponse(content="", tool_call=LLMToolCall(name="current_time", arguments={})),
            LLMResponse(content="The time is enough."),
            LLMResponse(content="Current result based on web search."),
        ]
    )
    web_search = FakeWebSearchTool()
    agent = ResearchAgent(provider, ToolRegistry([FakeCurrentTimeTool(), web_search]))

    result = agent.run("What is the latest NVIDIA news today?")

    assert web_search.calls == [{"query": "What is the latest NVIDIA news today?", "max_results": 5}]
    assert result.tool_name == "web_search"
    assert "Current result based on web search." in result.answer
    assert "Sources:" in result.answer


def test_research_agent_returns_actionable_error_when_search_tool_fails() -> None:
    class FailingWebSearchTool(FakeWebSearchTool):
        def execute(self, arguments: dict) -> ToolResult:
            self.calls.append(arguments)
            return ToolResult(content="Error: TAVILY_API_KEY is not configured in settings or environment.")

    provider = ScriptedProvider([LLMResponse(content="I cannot browse from here.")])
    web_search = FailingWebSearchTool()
    agent = ResearchAgent(provider, ToolRegistry([web_search]))

    result = agent.run("Search the web")

    assert result.tool_name == "web_search"
    assert "TAVILY_API_KEY is not configured" in result.answer
    assert len(provider.calls) == 1
