from typing import TypedDict, Sequence, List
from langchain_core.messages import BaseMessage

class Post(TypedDict):
    """Type definition for the LinkedIn post."""
    day: str
    article_id: str
    post_content: str

class PostAgentState(TypedDict):
    """Type definition for the LinkedIn agent state."""
    messages: Sequence[BaseMessage]
    next_step: str
    topic: str | None
    news_content: dict | None
    selected_news: dict | None
    linkedin_post: str | None

class ResearchAgentState(TypedDict):
    """Type definition for the LinkedIn agent state."""
    messages: Sequence[BaseMessage]
    next_step: str
    news_content: dict | None
    selected_news: List[dict] | None
    posting_plans: str

class PostingPlan(TypedDict):
    """Type definition for the LinkedIn posting plan."""
    day: str
    article_index: int
    title: str
    link: str
    snippet: str
    reason: str