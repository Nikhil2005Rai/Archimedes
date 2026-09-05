from typing import Literal, TypedDict

from langgraph.graph import END, StateGraph

from app.agents.planner import PlannerAgent, PlannerResult
from app.agents.registry import AgentBuildContext, AgentRegistry, build_agent_registry
from app.providers.base import LLMMessage, LLMProvider
from app.providers.embeddings.base import EmbeddingProvider
from app.retrieval.repository import RetrievalRepository
from app.tools.registry import ToolRegistry


_DISCLAIMER_MARKERS = (
    "don't have access to real-time",
    "cannot perform live",
    "cannot browse the web",
    "don't have the ability to perform",
    "trigger other agents",
    "no real-time data",
)

_WEB_REQUEST_MARKERS = (
    "web search",
    "search the web",
    "search online",
    "browse",
    "look up",
    "latest",
    "recent",
    "current",
    "today",
    "news",
    "live",
    "price",
    "weather",
    "stock",
)

_LOCAL_KNOWLEDGE_MARKERS = (
    "uploaded",
    "my document",
    "my docs",
    "my notes",
    "knowledge base",
    "rag",
)


def _looks_like_web_request(user_input: str) -> bool:
    text = user_input.lower()
    if any(marker in text for marker in _LOCAL_KNOWLEDGE_MARKERS):
        return False
    return any(marker in text for marker in _WEB_REQUEST_MARKERS)


class AgentGraphState(TypedDict, total=False):
    user_input: str
    history: list[LLMMessage]
    route: str
    answer: str
    thought_process: str | None
    tool_name: str | None
    tool_arguments: dict | None
    tool_output: str | None
    agent_name: str | None
    retrieval_query: str | None
    retrieval_chunk_ids: list[str] | None
    retrieval_scores: list[float] | None


class MultiAgentGraph(PlannerAgent):
    def __init__(
        self,
        llm_provider: LLMProvider,
        tools: ToolRegistry,
        agents: AgentRegistry | None = None,
        embedding_provider: EmbeddingProvider | None = None,
        retrieval_repository: RetrievalRepository | None = None,
        user_id: str | None = None,
        workspace_id: str | None = None,
    ) -> None:
        self.llm_provider = llm_provider
        self.tools = tools
        self.agents = agents if agents is not None else build_agent_registry()
        self.embedding_provider = embedding_provider
        self.retrieval_repository = retrieval_repository
        self.user_id = user_id
        self.workspace_id = workspace_id
        self.graph = self._build_graph()

    def run(self, user_input: str, history: list[LLMMessage] | None = None) -> PlannerResult:
        final_state = self.graph.invoke({"user_input": user_input, "history": history or []})
        return PlannerResult(
            answer=final_state.get("answer", ""),
            tool_name=final_state.get("tool_name"),
            tool_arguments=final_state.get("tool_arguments"),
            tool_output=final_state.get("tool_output"),
            agent_name=final_state.get("agent_name"),
            retrieval_query=final_state.get("retrieval_query"),
            retrieval_chunk_ids=final_state.get("retrieval_chunk_ids"),
            retrieval_scores=final_state.get("retrieval_scores"),
            thought_process=final_state.get("thought_process"),
        )

    def _build_graph(self):
        graph = StateGraph(AgentGraphState)
        graph.add_node("planner", self._planner_node)
        graph.add_node("research", self._research_node)
        graph.add_node("knowledge", self._knowledge_node)
        graph.add_node("coding", self._coding_node)
        graph.add_node("data_analyst", self._data_analyst_node)
        graph.add_node("security_auditor", self._security_auditor_node)
        graph.add_node("devops", self._devops_node)
        graph.add_node("multi", self._multi_node)
        graph.set_entry_point("planner")
        graph.add_conditional_edges(
            "planner",
            self._route_from_planner,
            {
                "research": "research",
                "knowledge": "knowledge",
                "coding": "coding",
                "data_analyst": "data_analyst",
                "security_auditor": "security_auditor",
                "devops": "devops",
                "multi": "multi",
                "end": END,
            },
        )
        graph.add_edge("research", END)
        graph.add_edge("knowledge", END)
        graph.add_edge("coding", END)
        graph.add_edge("data_analyst", END)
        graph.add_edge("security_auditor", END)
        graph.add_edge("devops", END)
        graph.add_edge("multi", END)
        return graph.compile()

    def _planner_node(self, state: AgentGraphState) -> AgentGraphState:
        context_str = ""
        retrieval_chunk_ids = []
        retrieval_scores = []

        if self.embedding_provider and self.retrieval_repository and self.user_id:
            try:
                query_embedding = self.embedding_provider.embed([state["user_input"]])[0]
                chunks = self.retrieval_repository.search(
                    user_id=self.user_id, embedding=query_embedding, limit=2, workspace_id=self.workspace_id
                )
                relevant_chunks = [chunk for chunk in chunks if chunk.score <= 0.5]
                if relevant_chunks:
                    from app.providers.prompt_safety import wrap_untrusted_content
                    context_str = "\n\n".join(
                        wrap_untrusted_content(f"retrieved_chunk id={chunk.id} score={chunk.score:.4f}", chunk.content)
                        for chunk in relevant_chunks
                    )
                    retrieval_chunk_ids = [chunk.id for chunk in relevant_chunks]
                    retrieval_scores = [chunk.score for chunk in relevant_chunks]
            except Exception as exc:
                import logging
                logging.getLogger(__name__).warning(f"RAG embedding retrieval skipped due to error: {exc}")

        system_content = (
            "You are the Planner agent in Archimedes AI OS monolith. Decide which specialist agent should fulfill the request:\n"
            "- If user asks about data analysis, SQL queries, CSV datasets, or charts/graphs -> respond 'ROUTE: data_analyst'\n"
            "- If user asks about code security audits, OWASP vulnerabilities, or architecture diagrams -> respond 'ROUTE: security_auditor'\n"
            "- If user asks about Dockerfiles, Kubernetes, Terraform, or CI/CD pipelines -> respond 'ROUTE: devops'\n"
            "- If user asks specifically about writing, debugging, or executing code -> respond 'ROUTE: coding'\n"
            "- If the request asks you to SEARCH/FIND/GET live or recent data, even if it also asks for a table, graph, or report from that data -> respond 'ROUTE: research' (never answer directly about lacking real-time access; you have a research agent for exactly this).\n"
            "- If user asks about uploaded documents/notes -> respond 'ROUTE: knowledge'\n"
            "- If user asks for live web search, facts, news -> respond 'ROUTE: research'\n"
            "- If request combines code AND web search -> respond 'ROUTE: coding, research'\n"
            "- Otherwise answer directly prefixed with 'ANSWER:'.\n"
        )

        if context_str:
            system_content += f"\n\nRetrieved knowledge:\n{context_str}"

        messages = [
            LLMMessage(role="system", content=system_content.strip()),
            *state.get("history", [])[-12:],
            LLMMessage(role="user", content=state["user_input"]),
        ]
        response = self.llm_provider.generate(messages)
        content = response.content.strip()
        thought = response.thought

        import re
        if "<thought>" in content.lower():
            t_match = re.search(r"<thought>(.*?)</thought>", content, flags=re.DOTALL | re.IGNORECASE)
            if t_match and not thought:
                thought = t_match.group(1).strip()
            content = re.sub(r"<thought>.*?</thought>", "", content, flags=re.DOTALL | re.IGNORECASE).strip()

        content_lower = content.lower().strip()
        route = "end"
        if "route:" in content_lower:
            idx = content_lower.find("route:")
            route_part = content_lower[idx + 6:].strip()
            route_part = route_part.split("\n")[0].strip()
            route_part = route_part.rstrip(".!?,")

            if "coding" in route_part and "research" in route_part:
                route = "multi"
            elif "data_analyst" in route_part:
                route = "data_analyst"
            elif "security_auditor" in route_part:
                route = "security_auditor"
            elif "devops" in route_part:
                route = "devops"
            elif "coding" in route_part:
                route = "coding"
            elif "knowledge" in route_part:
                route = "knowledge"
            elif "research" in route_part:
                route = "research"

        if route != "end":
            return {"route": route, "thought_process": thought}

        if content_lower.startswith("answer:"):
            content = content.split(":", 1)[1].strip()

        if any(marker in content_lower for marker in _DISCLAIMER_MARKERS):
            return {"route": "research", "thought_process": thought}

        if _looks_like_web_request(state["user_input"]):
            return {"route": "research", "thought_process": thought}

        return {
            "route": "end",
            "answer": content,
            "agent_name": "planner",
            "thought_process": thought,
            "retrieval_query": state["user_input"] if context_str else None,
            "retrieval_chunk_ids": retrieval_chunk_ids if context_str else None,
            "retrieval_scores": retrieval_scores if context_str else None,
        }

    @staticmethod
    def _route_from_planner(state: AgentGraphState) -> Literal["research", "knowledge", "coding", "data_analyst", "security_auditor", "devops", "multi", "end"]:
        route = state.get("route")
        if route in ("multi", "coding", "research", "knowledge", "data_analyst", "security_auditor", "devops"):
            return route
        return "end"

    def _multi_node(self, state: AgentGraphState) -> AgentGraphState:
        user_input = state["user_input"]
        coding_state = {
            **state,
            "user_input": f"Fulfill ONLY code generation/execution: {user_input}",
        }
        research_state = {
            **state,
            "user_input": f"Fulfill ONLY web search/research: {user_input}",
        }

        coding_res = self._coding_node(coding_state)
        research_res = self._research_node(research_state)

        answer_parts = []
        if coding_res.get("answer"):
            answer_parts.append(f"### Code Execution & Solution\n{coding_res['answer']}")
        if research_res.get("answer"):
            answer_parts.append(f"### Web Research Results\n{research_res['answer']}")

        tools_used = [t for t in [coding_res.get("tool_name"), research_res.get("tool_name")] if t]

        return {
            "answer": "\n\n---\n\n".join(answer_parts),
            "tool_name": ", ".join(tools_used) if tools_used else None,
            "agent_name": "planner + coding + research",
            "thought_process": state.get("thought_process"),
        }

    def _research_node(self, state: AgentGraphState) -> AgentGraphState:
        research_agent = self.agents.build("research", self._agent_context())
        result = research_agent.run(state["user_input"], history=state.get("history", []))
        return {
            "answer": result.answer,
            "tool_name": result.tool_name,
            "tool_arguments": result.tool_arguments,
            "tool_output": result.tool_output,
            "agent_name": result.agent_name or "research",
        }

    def _knowledge_node(self, state: AgentGraphState) -> AgentGraphState:
        knowledge_agent = self.agents.build("knowledge", self._agent_context())
        result = knowledge_agent.run(state["user_input"], history=state.get("history", []))
        return {
            "answer": result.answer,
            "agent_name": result.agent_name or "knowledge",
            "retrieval_query": result.retrieval_query,
            "retrieval_chunk_ids": result.retrieval_chunk_ids,
            "retrieval_scores": result.retrieval_scores,
        }

    def _coding_node(self, state: AgentGraphState) -> AgentGraphState:
        coding_agent = self.agents.build("coding", self._agent_context())
        result = coding_agent.run(state["user_input"], history=state.get("history", []))
        return {
            "answer": result.answer,
            "tool_name": result.tool_name,
            "tool_arguments": result.tool_arguments,
            "tool_output": result.tool_output,
            "agent_name": result.agent_name or "coding",
        }

    def _data_analyst_node(self, state: AgentGraphState) -> AgentGraphState:
        agent = self.agents.build("data_analyst", self._agent_context())
        result = agent.run(state["user_input"], history=state.get("history", []))
        return {
            "answer": result.answer,
            "tool_name": result.tool_name,
            "tool_arguments": result.tool_arguments,
            "tool_output": result.tool_output,
            "agent_name": result.agent_name or "data_analyst",
        }

    def _security_auditor_node(self, state: AgentGraphState) -> AgentGraphState:
        agent = self.agents.build("security_auditor", self._agent_context())
        result = agent.run(state["user_input"], history=state.get("history", []))
        return {
            "answer": result.answer,
            "tool_name": result.tool_name,
            "tool_arguments": result.tool_arguments,
            "tool_output": result.tool_output,
            "agent_name": result.agent_name or "security_auditor",
        }

    def _devops_node(self, state: AgentGraphState) -> AgentGraphState:
        agent = self.agents.build("devops", self._agent_context())
        result = agent.run(state["user_input"], history=state.get("history", []))
        return {
            "answer": result.answer,
            "tool_name": result.tool_name,
            "tool_arguments": result.tool_arguments,
            "tool_output": result.tool_output,
            "agent_name": result.agent_name or "devops",
        }

    def _agent_context(self) -> AgentBuildContext:
        return AgentBuildContext(
            llm_provider=self.llm_provider,
            tools=self.tools,
            embedding_provider=self.embedding_provider,
            retrieval_repository=self.retrieval_repository,
            user_id=self.user_id,
            workspace_id=self.workspace_id,
        )
