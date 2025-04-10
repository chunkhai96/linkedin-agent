from langgraph.graph import StateGraph, END
from .types import AgentState
from .models import get_llm
from .prompts import analyze_news_prompt, generate_post_prompt
from langchain_community.tools import DuckDuckGoSearchResults
from .utils.linkedin_client import LinkedInClient
from .utils.output_parser import LinkedInPostOutputParser
import os
import json  # Add this at the top with other imports

class LinkedInPostAgent:
    def __init__(self, stream: bool = False):
        """Initialize the LinkedIn Post Agent with configurable streaming."""
        self.llm = get_llm(stream=stream)
        self.workflow = self._create_workflow()

    def _search_news(self, state: AgentState) -> AgentState:
        """Search for latest news based on the given topic."""
        print(f"[STEP 1/4] Searching for latest news about: {state['topic']}")
        search = DuckDuckGoSearchResults(output_format="list", max_results=20)
        topic = state["topic"]
        search_query = f"latest news about {topic} in the last week"
        news_content = search.invoke(search_query)
        # Convert list to JSON text
        news_json = json.dumps(news_content, indent=2)
        print(f"Found {len(news_content)} news items")
        
        return {
            "messages": state["messages"],
            "topic": topic,
            "news_content": {"raw_content": news_json},
            "next_step": "analyze"
        }

    def _analyze_news(self, state: AgentState) -> AgentState:
        """Analyze the news content and extract key insights."""
        print(f"[STEP 2/4] Analyzing news content about: {state['topic']}")
        messages = analyze_news_prompt.format_messages(messages=[{"role": "user", "content": state["news_content"]["raw_content"]}])
        response = self.llm.invoke(messages)
        print(f"Analysis complete. Key insights extracted")
        return {
            "messages": [*state["messages"], response],
            "news_content": state["news_content"],
            "topic": state["topic"],
            "next_step": "generate_post"
        }

    def _generate_post(self, state: AgentState) -> AgentState:
        """Generate a LinkedIn post based on the news analysis."""
        print(f"[STEP 3/4] Generating LinkedIn post about: {state['topic']}")
        messages = generate_post_prompt.format_messages(messages=state["messages"])
        response = self.llm.invoke(messages)
        post_content = response.content #LinkedInPostOutputParser.parse(response.content)
        if post_content is None:
            raise ValueError("Generated post content does not contain the required markers")
        print(f"Post draft generated successfully")
        return {
            "messages": [*state["messages"], response],
            "linkedin_post": post_content,
            "next_step": "post_to_linkedin"
        }

    def _post_to_linkedin(self, state: AgentState) -> AgentState:
        """Post the generated content to LinkedIn."""
        print(f"[STEP 4/4] Posting to LinkedIn about: {state['topic']}")
        access_token = os.getenv("LINKEDIN_ACCESS_TOKEN")
        if not access_token:
            raise ValueError("LinkedIn access token not found in environment variables")
        
        client = LinkedInClient(access_token)
        post_response = client.post_content(state["linkedin_post"])
        print(f"Post successfully published to LinkedIn")
        
        return {
            "messages": state["messages"],
            "linkedin_post": state["linkedin_post"],
            "post_response": post_response,
            "next_step": END
        }

    def _create_workflow(self) -> StateGraph:
        """Create and configure the news-to-LinkedIn post agent workflow."""
        workflow = StateGraph(AgentState)
        
        # Add nodes
        workflow.add_node("search", self._search_news)
        workflow.add_node("analyze", self._analyze_news)
        workflow.add_node("generate_post", self._generate_post)
        workflow.add_node("post_to_linkedin", self._post_to_linkedin)
        
        # Add edges
        workflow.add_edge("search", "analyze")
        workflow.add_edge("analyze", "generate_post")
        workflow.add_edge("generate_post", "post_to_linkedin")
        workflow.add_edge("post_to_linkedin", END)
        
        # Set entry point
        workflow.set_entry_point("search")
        
        return workflow.compile()

    def run(self, topic: str) -> AgentState:
        """Run the LinkedIn post agent workflow with the given topic."""
        initial_state = {
            "messages": [],
            "topic": topic,
            "next_step": "search"
        }
        return self.workflow.invoke(initial_state)

def create_linkedin_post_agent(stream: bool = False) -> StateGraph:
    """Create and return a LinkedIn post agent instance."""
    agent = LinkedInPostAgent(stream=stream)
    return agent.workflow