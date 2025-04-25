from langchain_core.messages import HumanMessage
from .agents.post_agent import create_linkedin_post_agent
from .agents.research_agent import LinkedInResearchAgent

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

def research_post(stream: bool = False) -> str:
    """Research for topics and create LinkedIn posting plan.
    
    Args:
        stream: Whether to stream the output step by step
        topics: The list of topics to research
        
    Returns:
        A list containing the topics to post in a week if stream is False,
        or a list of intermediate results if stream is True
    """
    agent = LinkedInResearchAgent(stream=stream)
    result = agent.run()
    if stream:
        return result["messages"]
    return result.get("posts", [])