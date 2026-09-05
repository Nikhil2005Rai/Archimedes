from app.agents.graph import MultiAgentGraph
from app.agents.registry import build_agent_registry
from app.domain.entities import RetrievedChunk
from app.providers.base import LLMMessage, LLMResponse, ToolSchema
from app.tools.registry import ToolRegistry


class ScriptedGraphProvider:
    def __init__(self, responses: list[LLMResponse]) -> None:
        self.responses = responses
        self.calls: list[tuple[list[LLMMessage], list[ToolSchema] | None]] = []

    def generate(self, messages: list[LLMMessage], tools: list[ToolSchema] | None = None) -> LLMResponse:
        self.calls.append((messages, tools))
        return self.responses.pop(0)


class FakeEmbeddingProvider:
    def embed(self, texts: list[str]) -> list[list[float]]:
        return [[0.1] * 768 for _ in texts]


class FakeRetrievalRepository:
    def __init__(self) -> None:
        self.search_workspace_id = None

    def search(self, user_id: str, embedding: list[float], limit: int = 4, workspace_id: str | None = None) -> list[RetrievedChunk]:
        self.search_workspace_id = workspace_id
        return [
            RetrievedChunk(
                id="chunk-1",
                document_id="document-1",
                content="uploaded project notes",
                score=0.01,
            )
        ]


def test_graph_planner_answers_directly() -> None:
    provider = ScriptedGraphProvider([LLMResponse(content="ANSWER: Direct graph answer")])
    graph = MultiAgentGraph(llm_provider=provider, tools=ToolRegistry([]), agents=build_agent_registry())

    result = graph.run("Say hello")

    assert result.answer == "Direct graph answer"
    assert result.agent_name == "planner"
    assert len(provider.calls) == 1


def test_graph_routes_to_research_agent() -> None:
    provider = ScriptedGraphProvider(
        [
            LLMResponse(content="ROUTE: research"),
            LLMResponse(content="Research answer"),
        ]
    )
    graph = MultiAgentGraph(llm_provider=provider, tools=ToolRegistry([]), agents=build_agent_registry())

    result = graph.run("Research this")

    assert result.answer == "Research answer"
    assert result.agent_name == "research"
    assert len(provider.calls) == 2


def test_graph_routes_disclaimer_shaped_planner_fallback_to_research_agent() -> None:
    provider = ScriptedGraphProvider(
        [
            LLMResponse(content="I don't have access to real-time data, so I cannot browse the web for that."),
            LLMResponse(content="Research agent handled the live-data request."),
        ]
    )
    graph = MultiAgentGraph(llm_provider=provider, tools=ToolRegistry([]), agents=build_agent_registry())

    result = graph.run("Search the latest water level and plot a graph")

    assert result.answer == "Research agent handled the live-data request."
    assert result.agent_name == "research"
    assert len(provider.calls) == 2


def test_graph_routes_live_web_intent_to_research_even_if_planner_answers_directly() -> None:
    provider = ScriptedGraphProvider(
        [
            LLMResponse(content="ANSWER: I will answer from memory."),
            LLMResponse(content="Research agent handled the latest request."),
        ]
    )
    graph = MultiAgentGraph(llm_provider=provider, tools=ToolRegistry([]), agents=build_agent_registry())

    result = graph.run("What is the latest NVIDIA news today?")

    assert result.answer == "Research agent handled the latest request."
    assert result.agent_name == "research"
    assert len(provider.calls) == 2


def test_graph_routes_to_coding_agent() -> None:
    provider = ScriptedGraphProvider(
        [
            LLMResponse(content="ROUTE: coding"),
            LLMResponse(content="Here is the python code: print('test')"),
        ]
    )
    graph = MultiAgentGraph(llm_provider=provider, tools=ToolRegistry([]), agents=build_agent_registry())

    result = graph.run("Write a python script to test")

    assert result.answer == "Here is the python code: print('test')"
    assert result.agent_name == "coding"
    assert len(provider.calls) == 2


def test_graph_routes_to_knowledge_agent() -> None:
    provider = ScriptedGraphProvider(
        [
            LLMResponse(content="ROUTE: knowledge"),
            LLMResponse(content="Grounded knowledge answer"),
        ]
    )
    retrieval_repository = FakeRetrievalRepository()
    graph = MultiAgentGraph(
        llm_provider=provider,
        tools=ToolRegistry([]),
        agents=build_agent_registry(),
        embedding_provider=FakeEmbeddingProvider(),
        retrieval_repository=retrieval_repository,
        user_id="user-1",
        workspace_id="workspace-1",
    )

    result = graph.run("Use my uploaded knowledge")

    assert result.answer == "Grounded knowledge answer"
    assert result.agent_name == "knowledge"
    assert result.retrieval_chunk_ids == ["chunk-1"]
    assert retrieval_repository.search_workspace_id == "workspace-1"
    assert len(provider.calls) == 2

    # Verify the Knowledge agent's system message wraps chunks in delimiters
    knowledge_call_messages = provider.calls[1][0]
    system_msg = knowledge_call_messages[0]
    assert system_msg.role == "system"
    assert "<retrieved_chunk id=chunk-1 score=0.0100>" in system_msg.content
    assert "</retrieved_chunk id=chunk-1 score=0.0100>" in system_msg.content
    assert "uploaded project notes" in system_msg.content
def test_graph_planner_uses_injected_knowledge() -> None:
    # Planner decides it can answer directly using the injected knowledge
    provider = ScriptedGraphProvider([LLMResponse(content="ANSWER: I found this in your notes: uploaded project notes")])
    graph = MultiAgentGraph(
        llm_provider=provider,
        tools=ToolRegistry([]),
        agents=build_agent_registry(),
        embedding_provider=FakeEmbeddingProvider(),
        retrieval_repository=FakeRetrievalRepository(),
        user_id="user-1",
    )

    result = graph.run("What are my notes about?")

    assert result.answer == "I found this in your notes: uploaded project notes"
    assert result.agent_name == "planner"
    assert result.retrieval_chunk_ids == ["chunk-1"]
    assert len(provider.calls) == 1

    # Verify the planner's system message includes the injected context
    planner_call_messages = provider.calls[0][0]
    system_msg = planner_call_messages[0]
    assert system_msg.role == "system"
    assert "<retrieved_chunk id=chunk-1 score=0.0100>" in system_msg.content
    assert "uploaded project notes" in system_msg.content


def test_graph_planner_does_not_misroute_coincidental_keywords() -> None:
    provider = ScriptedGraphProvider([LLMResponse(content="ANSWER: I will help you with security auditor and devops infrastructure.")])
    graph = MultiAgentGraph(llm_provider=provider, tools=ToolRegistry([]), agents=build_agent_registry())

    result = graph.run("Explain how security auditor works")

    assert result.answer == "I will help you with security auditor and devops infrastructure."
    assert result.agent_name == "planner"
    assert len(provider.calls) == 1
