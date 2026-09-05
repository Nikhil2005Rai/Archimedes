import httpx

from app.core.config import settings
from app.tools.base import ToolResult

TAVILY_API_URL = "https://api.tavily.com/search"


class TavilySearchTool:
    name = "web_search"
    description = (
        "Searches the live web for real-time information, current facts, news, and live web content. "
        "Use this tool when answering questions about recent events, current weather, stock prices, or up-to-date documentation."
    )
    parameters = {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "The web search query."},
            "max_results": {"type": "integer", "description": "Maximum number of search results to return (default 5).", "default": 5},
        },
        "required": ["query"],
    }

    def execute(self, arguments: dict) -> ToolResult:
        result_text = self.run(arguments)
        return ToolResult(content=result_text)

    def run(self, arguments: dict) -> str:
        query = str(arguments.get("query", ""))
        try:
            max_results = int(arguments.get("max_results", 5))
        except (TypeError, ValueError):
            max_results = 5
        max_results = min(max(max_results, 1), 10)

        if not query.strip():
            return "Error: Search query cannot be empty."

        api_key = settings.tavily_api_key
        if not api_key:
            return "Error: TAVILY_API_KEY is not configured in settings or environment."

        payload = {
            "api_key": api_key,
            "query": query,
            "max_results": max_results,
            "search_depth": "basic",
            "include_answer": True,
        }

        try:
            response = httpx.post(TAVILY_API_URL, json=payload, timeout=15)
            response.raise_for_status()
        except httpx.TimeoutException:
            return "Error: Tavily web search timed out."
        except httpx.HTTPStatusError as exc:
            return f"Error: Tavily search returned {exc.response.status_code} — {exc.response.text[:300]}"
        except httpx.HTTPError as exc:
            return f"Error: Could not reach Tavily search service ({exc})."

        data = response.json()
        answer = data.get("answer")
        results = data.get("results", [])

        parts = []
        if answer:
            parts.append(f"Summary Answer: {answer}")

        if results:
            parts.append("Web Search Results:")
            for idx, item in enumerate(results, 1):
                title = item.get("title", "No Title")
                url = item.get("url", "")
                content = item.get("content", "")
                parts.append(f"{idx}. [{title}]({url})\n{content}")

        if not parts:
            return "No web search results found for query."

        return "\n\n".join(parts)
