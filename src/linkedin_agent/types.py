from typing import TypedDict, Sequence
from langchain_core.messages import BaseMessage

class AgentState(TypedDict):
    """Type definition for the LinkedIn agent state."""
    messages: Sequence[BaseMessage]
    next_step: str
    topic: str | None
    news_content: dict | None
    linkedin_post: str | None