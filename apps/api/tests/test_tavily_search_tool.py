from unittest.mock import MagicMock, patch

import httpx

from app.core.config import settings
from app.tools.tavily_search import TavilySearchTool


def test_tavily_search_missing_api_key() -> None:
    tool = TavilySearchTool()
    with patch.object(settings, "tavily_api_key", ""):
        result = tool.run({"query": "latest news"})
        assert "Error: TAVILY_API_KEY is not configured" in result


def test_tavily_search_empty_query() -> None:
    tool = TavilySearchTool()
    result = tool.run({"query": "  "})
    assert "Error: Search query cannot be empty." in result


def test_tavily_search_success() -> None:
    tool = TavilySearchTool()
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "answer": "FastAPI is a modern web framework for Python.",
        "results": [
            {
                "title": "FastAPI Documentation",
                "url": "https://fastapi.tiangolo.com",
                "content": "FastAPI framework, high performance, easy to learn.",
            }
        ],
    }

    with patch.object(settings, "tavily_api_key", "tvly-test-key"):
        with patch("httpx.post", return_value=mock_resp) as mock_post:
            result = tool.run({"query": "FastAPI docs", "max_results": 1})
            mock_post.assert_called_once()
            assert "Summary Answer: FastAPI is a modern web framework" in result
            assert "[FastAPI Documentation](https://fastapi.tiangolo.com)" in result
            assert "FastAPI framework, high performance" in result


def test_tavily_search_normalizes_max_results() -> None:
    tool = TavilySearchTool()
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {"results": []}

    with patch.object(settings, "tavily_api_key", "tvly-test-key"):
        with patch("httpx.post", return_value=mock_resp) as mock_post:
            tool.run({"query": "FastAPI docs", "max_results": "20"})

    payload = mock_post.call_args.kwargs["json"]
    assert payload["max_results"] == 10


def test_tavily_search_timeout() -> None:
    tool = TavilySearchTool()
    with patch.object(settings, "tavily_api_key", "tvly-test-key"):
        with patch("httpx.post", side_effect=httpx.TimeoutException("Timeout")):
            result = tool.run({"query": "test query"})
            assert "Error: Tavily web search timed out." in result


def test_tavily_search_http_error() -> None:
    tool = TavilySearchTool()
    req = httpx.Request("POST", "https://api.tavily.com/search")
    res = httpx.Response(401, text="Unauthorized key", request=req)
    err = httpx.HTTPStatusError("401 error", request=req, response=res)

    with patch.object(settings, "tavily_api_key", "invalid-key"):
        with patch("httpx.post", side_effect=err):
            result = tool.run({"query": "test query"})
            assert "Error: Tavily search returned 401" in result
