import re


_MARKDOWN_LINK_RE = re.compile(r"\[([^\]]+)\]\((https?://[^)\s]+)\)")
_SOURCES_HEADING_RE = re.compile(r"(^|\n)#{0,3}\s*sources\s*:?", re.IGNORECASE)
_WEB_SEARCH_TOOL_NAMES = ("web_search", "google_search", "search_web", "internet_search", "tavily_search", "bing_search")


def append_web_sources(answer: str, tool_name: str | None, tool_output: str | None, max_sources: int = 5) -> str:
    if not tool_name or not any(name in tool_name for name in _WEB_SEARCH_TOOL_NAMES) or not tool_output:
        return answer
    if _SOURCES_HEADING_RE.search(answer):
        return answer

    sources = _extract_sources(tool_output, max_sources=max_sources)
    if not sources:
        return answer

    source_lines = ["Sources:"]
    for index, (title, url) in enumerate(sources, 1):
        source_lines.append(f"{index}. [{title}]({url})")
    return f"{answer.rstrip()}\n\n" + "\n".join(source_lines)


def _extract_sources(tool_output: str, max_sources: int) -> list[tuple[str, str]]:
    sources: list[tuple[str, str]] = []
    seen_urls: set[str] = set()
    for title, url in _MARKDOWN_LINK_RE.findall(tool_output):
        clean_title = _clean_title(title)
        clean_url = url.rstrip(".,;")
        if not clean_title or clean_url in seen_urls:
            continue
        seen_urls.add(clean_url)
        sources.append((clean_title, clean_url))
        if len(sources) >= max_sources:
            break
    return sources


def _clean_title(title: str) -> str:
    return re.sub(r"\s+", " ", title).strip()
