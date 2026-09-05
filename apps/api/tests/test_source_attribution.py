from app.agents.source_attribution import append_web_sources


def test_append_web_sources_extracts_unique_markdown_links() -> None:
    tool_output = (
        "Web Search Results:\n"
        "1. [First Source](https://example.com/a)\nAlpha\n\n"
        "2. [Second Source](https://example.com/b)\nBeta\n\n"
        "3. [First Source](https://example.com/a)\nDuplicate"
    )

    answer = append_web_sources("Final answer.", "web_search", tool_output)

    assert answer == (
        "Final answer.\n\n"
        "Sources:\n"
        "1. [First Source](https://example.com/a)\n"
        "2. [Second Source](https://example.com/b)"
    )


def test_append_web_sources_does_not_duplicate_existing_sources() -> None:
    answer = "Final answer.\n\nSources:\n1. [Already there](https://example.com)"
    tool_output = "1. [First Source](https://example.com/a)\nAlpha"

    assert append_web_sources(answer, "web_search", tool_output) == answer


def test_append_web_sources_ignores_non_web_tools() -> None:
    tool_output = "1. [First Source](https://example.com/a)\nAlpha"

    assert append_web_sources("Final answer.", "execute_code", tool_output) == "Final answer."
