from unittest.mock import MagicMock
from app.tools.web_reader import WebReaderTool
from app.tools.chart_generator import ChartGeneratorTool
from app.tools.github_inspector import GithubInspectorTool
from app.tools.db_inspector import DbInspectorTool
from app.tools.registry import ToolRegistry, build_tool_registry
from app.agents.data_analyst import DataAnalystAgent
from app.agents.security_auditor import SecurityAuditorAgent
from app.agents.devops_agent import DevOpsAgent
from app.agents.registry import build_agent_registry, AgentBuildContext
from app.providers.base import LLMResponse, LLMToolCall, LLMMessage
from app.tools.base import ToolResult


class FakeLLMProvider:
    def __init__(self, response_text: str = "Test answer", tool_call=None):
        self.response_text = response_text
        self.tool_call = tool_call

    def generate(self, messages: list[LLMMessage], tools=None) -> LLMResponse:
        return LLMResponse(content=self.response_text, tool_call=self.tool_call)


def test_web_reader_tool():
    tool = WebReaderTool()
    assert tool.name == "web_reader"
    res = tool.run({"url": "invalid-scheme"})
    assert "Error:" in res


def test_chart_generator_tool():
    tool = ChartGeneratorTool()
    assert tool.name == "chart_generator"
    res = tool.run({
        "chart_type": "bar",
        "title": "Sales 2026",
        "data": [{"month": "Jan", "sales": 100}],
        "x_key": "month",
        "y_keys": ["sales"],
    })
    assert "```json:chart" in res
    assert "Sales 2026" in res


def test_github_inspector_tool():
    tool = GithubInspectorTool()
    assert tool.name == "github_inspector"
    res = tool.run({"owner": "", "repo": "", "action": "list_files"})
    assert "Error:" in res


def test_db_inspector_tool(db_session):
    tool = DbInspectorTool(session=db_session)
    assert tool.name == "db_inspector"
    res = tool.run({"action": "list_tables"})
    assert "user" in res.lower()
    assert "conversations" in res.lower()


def test_tool_registry_includes_new_tools():
    registry = build_tool_registry()
    names = [t.name for t in registry.all()]
    assert "web_reader" in names
    assert "chart_generator" in names
    assert "github_inspector" in names
    assert "db_inspector" in names


def test_new_agents():
    tools = build_tool_registry()
    provider = FakeLLMProvider("Data analysis complete.")
    ctx = AgentBuildContext(llm_provider=provider, tools=tools)
    registry = build_agent_registry()

    analyst = registry.build("data_analyst", ctx)
    res1 = analyst.run("Analyze sales data")
    assert res1.answer == "Data analysis complete."
    assert res1.agent_name == "data_analyst"

    security = registry.build("security_auditor", ctx)
    res2 = security.run("Audit C++ code")
    assert res2.agent_name == "security_auditor"

    devops = registry.build("devops", ctx)
    res3 = devops.run("Generate Dockerfile")
    assert res3.agent_name == "devops"


class ScriptedLLMProvider(FakeLLMProvider):
    def __init__(self, responses: list[LLMResponse]) -> None:
        self.responses = responses
        self.calls = []

    def generate(self, messages: list[LLMMessage], tools=None) -> LLMResponse:
        self.calls.append(messages)
        return self.responses.pop(0)


def test_new_agents_prompt_safety():
    first_resp = LLMResponse(
        content="",
        tool_call=LLMToolCall(
            name="github_inspector",
            arguments={"owner": "octocat", "repo": "hello", "action": "list_files"},
        ),
    )
    final_resp = LLMResponse(content="Final agent answer")

    llm_devops = ScriptedLLMProvider([first_resp, final_resp])
    tools = build_tool_registry()
    ctx = AgentBuildContext(llm_provider=llm_devops, tools=tools)
    registry = build_agent_registry()
    devops_agent = registry.build("devops", ctx)
    
    github_tool = tools.get("github_inspector")
    github_tool.run = lambda args: "Mocked GitHub Files List"

    res = devops_agent.run("Audit my devops pipeline")
    assert res.answer == "Final agent answer"
    assert len(llm_devops.calls) == 2
    follow_up_user_msg = llm_devops.calls[1][-1].content
    assert "<tool_output>" in follow_up_user_msg
    assert "</tool_output>" in follow_up_user_msg
    assert "Mocked GitHub Files List" in follow_up_user_msg

    first_resp_sec = LLMResponse(
        content="",
        tool_call=LLMToolCall(
            name="github_inspector",
            arguments={"owner": "octocat", "repo": "hello", "action": "list_files"},
        ),
    )
    final_resp_sec = LLMResponse(content="Final security answer")

    llm_security = ScriptedLLMProvider([first_resp_sec, final_resp_sec])
    ctx_sec = AgentBuildContext(llm_provider=llm_security, tools=tools)
    security_agent = registry.build("security_auditor", ctx_sec)

    res_sec = security_agent.run("Audit code security")
    assert res_sec.answer == "Final security answer"
    assert len(llm_security.calls) == 2
    follow_up_user_msg_sec = llm_security.calls[1][-1].content
    assert "<tool_output>" in follow_up_user_msg_sec
    assert "</tool_output>" in follow_up_user_msg_sec
    assert "Mocked GitHub Files List" in follow_up_user_msg_sec


class FakeTool:
    def __init__(self, name: str, content: str) -> None:
        self.name = name
        self.description = f"{name} test tool"
        self.parameters = {"type": "object", "properties": {}}
        self.content = content
        self.calls = []

    def execute(self, arguments: dict) -> ToolResult:
        self.calls.append(arguments)
        return ToolResult(content=self.content)


def test_data_analyst_chains_web_search_to_chart_generator_and_extracts_chart_block():
    web_search = FakeTool(
        "web_search",
        "Web Search Results:\n1. [Hydrology Report](https://example.com/hydro)\nLatest levels: Jan=10, Feb=12.",
    )
    chart_generator = FakeTool(
        "chart_generator",
        '```json:chart\n{"chart_type":"line","title":"Water levels","data":[{"month":"Jan","level":10}],"x_key":"month","y_keys":["level"]}\n```\n\nGenerated LINE chart: **Water levels** with 1 data points.',
    )
    tools = ToolRegistry([web_search, chart_generator])
    provider = ScriptedLLMProvider(
        [
            LLMResponse(
                content="",
                tool_call=LLMToolCall(name="web_search", arguments={"query": "latest water levels"}),
            ),
            LLMResponse(
                content="",
                tool_call=LLMToolCall(
                    name="chart_generator",
                    arguments={
                        "chart_type": "line",
                        "title": "Water levels",
                        "data": [{"month": "Jan", "level": 10}],
                        "x_key": "month",
                        "y_keys": ["level"],
                    },
                ),
            ),
            LLMResponse(content="Here is the requested chart."),
        ]
    )

    analyst = DataAnalystAgent(provider, tools)
    result = analyst.run("Search latest water levels and plot a graph")

    assert result.tool_name == "web_search, chart_generator"
    assert web_search.calls == [{"query": "latest water levels"}]
    assert chart_generator.calls[0]["title"] == "Water levels"
    assert result.answer.startswith("```json:chart\n")
    assert "Generated LINE chart" not in result.answer
    assert "Here is the requested chart." in result.answer
    assert "Sources:" in result.answer
    assert "[Hydrology Report](https://example.com/hydro)" in result.answer


def test_devops_agent_appends_sources_for_web_search_tool():
    web_search = FakeTool(
        "web_search",
        "Web Search Results:\n1. [Kubernetes Docs](https://kubernetes.io/docs/home/)\nCurrent docs.",
    )
    tools = ToolRegistry([web_search])
    provider = ScriptedLLMProvider(
        [
            LLMResponse(content="", tool_call=LLMToolCall(name="web_search", arguments={"query": "kubernetes docs"})),
            LLMResponse(content="Use the current Kubernetes guidance."),
        ]
    )

    agent = DevOpsAgent(provider, tools)
    result = agent.run("Find current Kubernetes deployment guidance")

    assert result.tool_name == "web_search"
    assert result.answer.endswith("1. [Kubernetes Docs](https://kubernetes.io/docs/home/)")
