from typing import List
from pydantic import BaseModel, Field


class SearchQueryList(BaseModel):
    query: List[str] = Field(
        description="A list of search queries to be used for web research."
    )
    rationale: str = Field(
        description="A brief explanation of why these queries are relevant to the research topic."
    )


class Reflection(BaseModel):
    is_sufficient: bool = Field(
        description="Whether the provided summaries are sufficient to answer the user's question."
    )
    knowledge_gap: str = Field(
        description="A description of what information is missing or needs clarification."
    )
    follow_up_queries: List[str] = Field(
        description="A list of follow-up queries to address the knowledge gap."
    )


class WebSource(BaseModel):
    title: str = Field(description="Title of the source")
    url: str = Field(description="URL of the source")
    snippet: str = Field(description="Short snippet or excerpt from the source")


class WebSearchResult(BaseModel):
    summary: str = Field(description="A synthesized summary of search findings")
    sources: List[WebSource] = Field(description="List of sources used for the summary")
