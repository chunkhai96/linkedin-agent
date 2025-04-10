from langchain_core.messages import HumanMessage
from .agents import create_linkedin_post_agent

def create_post(topic: str, stream: bool = False) -> str:
    """Process news content and generate a LinkedIn post.
    
    Args:
        topic: The topic of the news article
        stream: Whether to stream the output step by step
        
    Returns:
        A string containing the generated LinkedIn post if stream is False,
        or a list of intermediate results if stream is True
    """
    agent = create_linkedin_post_agent(stream=stream)
    result = agent.invoke({
        "messages": [HumanMessage(content=str(topic))],
        "next_step": "analyze_news",
        "topic": topic
    })
    if stream:
        return result["messages"]
    return result.get("linkedin_post", "")