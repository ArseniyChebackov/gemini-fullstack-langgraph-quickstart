"""Local directory search module for offline research without API calls.

This module provides functionality to search through local files instead of using
web search APIs, enabling offline operation and reducing API costs.
"""

import os
from pathlib import Path
from typing import Optional

from agent.tools_and_schemas import WebSearchResult, WebSource


def search_local_directory(
    search_query: str, directory: str, max_results: int = 5
) -> WebSearchResult:
    """Search local files for content matching the search query.

    Reads all text files in the specified directory and returns files
    whose content matches the search terms, along with extracted snippets.

    Args:
        search_query: Search terms to look for in files
        directory: Path to directory containing files to search
        max_results: Maximum number of sources to return

    Returns:
        WebSearchResult with summary and sources found in local files
    """
    if not os.path.isdir(directory):
        return WebSearchResult(
            summary=f"Error: Directory '{directory}' not found.",
            sources=[],
        )

    search_terms = search_query.lower().split()
    matching_sources = []
    file_contents = []

    # Read all text files in directory
    for file_path in Path(directory).rglob("*.txt"):
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
                file_contents.append((file_path, content))
        except Exception:
            continue
    
    # Also read markdown files
    for file_path in Path(directory).rglob("*.md"):
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
                file_contents.append((file_path, content))
        except Exception:
            continue

    if not file_contents:
        return WebSearchResult(
            summary=f"No text or markdown files found in directory '{directory}'.",
            sources=[],
        )

    # Search for matching files
    for file_path, content in file_contents:
        # Check if any search terms match
        matches = sum(1 for term in search_terms if term in content.lower())
        if matches > 0:
            # Extract snippet around first match
            lower_content = content.lower()
            snippet_start = 0
            for term in search_terms:
                idx = lower_content.find(term)
                if idx != -1:
                    snippet_start = max(0, idx - 50)
                    break

            snippet_end = min(len(content), snippet_start + 200)
            snippet = content[snippet_start:snippet_end].strip()

            matching_sources.append(
                WebSource(
                    title=file_path.name,
                    url=str(file_path),
                    snippet=snippet,
                )
            )

    # Sort by relevance (number of matching terms) and limit results
    matching_sources = matching_sources[:max_results]

    if not matching_sources:
        summary = (
            f"No files found in '{directory}' matching '{search_query}'."
        )
    else:
        # Generate summary from found content
        summaries = []
        for source in matching_sources:
            summaries.append(
                f"- {source.title}: {source.snippet[:100]}..."
            )
        summary = f"Found {len(matching_sources)} file(s) matching '{search_query}':\n" + "\n".join(
            summaries
        )

    return WebSearchResult(summary=summary, sources=matching_sources)
