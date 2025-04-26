import os
import json  # Add this at the top with other imports

from langgraph.graph import StateGraph, END
from langchain_community.tools import DuckDuckGoSearchResults

from ..types import PostAgentState
from ..models import get_llm
from ..prompts import analyze_news_prompt, generate_post_prompt, rank_news_prompt
from ..utils.linkedin_client import LinkedInClient
from ..utils.output_parser import LinkedInPostOutputParser


class LinkedInPostAgent:
    """Agent for automating the process of generating and posting LinkedIn content based on news topics."""
    def __init__(self, stream: bool = False, verbose: int = 1) -> None:
        """
        Initialize the LinkedIn Post Agent with configurable streaming and verbosity.

        Args:
            stream (bool): Whether to stream LLM responses.
            verbose (int): Controls logging output (0 = silent, 1 = normal).
        """
        self.llm = get_llm(stream=stream)
        self.verbose = verbose
        self.workflow = self._create_workflow()

    def _search_news(self, state: PostAgentState) -> PostAgentState:
        """
        Search for latest news based on the given topic.

        Args:
            state (PostAgentState): The current state containing the topic and messages.

        Returns:
            PostAgentState: Updated state with news content.
        """
        if self.verbose > 0:
            print(f"[STEP 1/5] Searching for latest news about: {state['topic']}")
        search = DuckDuckGoSearchResults(output_format="list", max_results=20)
        topic = state["topic"]
        search_query = f"latest news about {topic} in the last week"
        news_content = search.invoke(search_query)
        # Convert list to JSON text
        news_json = json.dumps(news_content, indent=2)
        if self.verbose > 0:
            print(f"Found {len(news_content)} news items")
        return {
            "messages": state["messages"],
            "topic": topic,
            "news_content": {"raw_content": news_json},
            "selected_news": None,
            "linkedin_post": None,
            "next_step": "analyze"
        }

    def _analyze_news(self, state: PostAgentState) -> PostAgentState:
        """
        Analyze the news content and extract key insights.

        Args:
            state (PostAgentState): The current state containing news content.

        Returns:
            PostAgentState: Updated state with analysis results.
        """
        if self.verbose > 0:
            print(f"[STEP 2/5] Analyzing news content about: {state['topic']}")
        messages = analyze_news_prompt.format_messages(messages=[{"role": "user", "content": state["news_content"]["raw_content"]}])
        response = self.llm.invoke(messages)
        if self.verbose > 0:
            print(f"Analysis complete. Key insights extracted")
        return {
            "messages": [*state["messages"], response],
            "news_content": state["news_content"],
            "topic": state["topic"],
            "selected_news": None,
            "linkedin_post": None,
            "next_step": "select_news"
        }

    def _select_news(self, state: PostAgentState) -> PostAgentState:
        """
        Select the most relevant news item based on the generated post.

        Args:
            state (PostAgentState): The current state containing analyzed news content.

        Returns:
            PostAgentState: Updated state with the selected news item.
        """
        if self.verbose > 0:
            print(f"[STEP 3/5] Selecting most relevant news item")
        news_items = state["news_content"]["raw_content"]

        messages = rank_news_prompt.format_messages(messages=[{"role": "user", "content": news_items}])

        response = self.llm.invoke(messages)
        
        try:
            most_relevant_index = int(response.content.strip())
            most_relevant_item = json.loads(news_items)[most_relevant_index]
        except (ValueError, IndexError):
            # Fallback to first item if parsing fails
            most_relevant_item = json.loads(news_items)[0]
        
        return {
            "messages": [*state["messages"], response],
            "news_content": state["news_content"],
            "topic": state["topic"],
            "selected_news": {
                "title": most_relevant_item.get("title", ""),
                "link": most_relevant_item.get("link", ""),
                "snippet": most_relevant_item.get("snippet", "")
            },
            "linkedin_post": None,
            "next_step": "generate_post"
        }

    def _generate_post(self, state: PostAgentState) -> PostAgentState:
        """
        Generate a LinkedIn post based on the news analysis.

        Args:
            state (PostAgentState): The current state containing the selected news item.

        Returns:
            PostAgentState: Updated state with the generated LinkedIn post.
        """
        if self.verbose > 0:
            print(f"[STEP 4/5] Generating LinkedIn post about: {state['topic']}")
        messages = generate_post_prompt.format_messages(messages=state["messages"])
        response = self.llm.invoke(messages)
        post_content = LinkedInPostOutputParser.parse(response.content)
        post_content = post_content + f"\n\n{state['selected_news']['link']}"
        if post_content is None:
            raise ValueError("Generated post content does not contain the required markers")
        if self.verbose > 0:
            print(f"Post draft generated successfully: \n")
            print(post_content)
        return {
            "messages": [*state["messages"], response],
            "linkedin_post": post_content,
            "next_step": "post_to_linkedin"
        }

    def _post_to_linkedin(self, state: PostAgentState) -> PostAgentState:
        """
        Post the generated content to LinkedIn.

        Args:
            state (PostAgentState): The current state containing the LinkedIn post content.

        Returns:
            PostAgentState: Updated state after posting to LinkedIn.
        """
        if self.verbose > 0:
            print(f"[STEP 5/5] Posting to LinkedIn about: {state['topic']}")
        access_token = os.getenv("LINKEDIN_ACCESS_TOKEN")
        if not access_token:
            raise ValueError("LinkedIn access token not found in environment variables")
        
        client = LinkedInClient(access_token)
        post_response = client.post_content(state["linkedin_post"])
        if self.verbose > 0:
            print(f"Post successfully published to LinkedIn. ({post_response})")
        
        return {
            "messages": state["messages"],
            "news_content": state["news_content"],
            "topic": state["topic"],
            "selected_news": state.get("selected_news"),
            "linkedin_post": state["linkedin_post"],
            "post_response": post_response,
            "next_step": END
        }

    def _create_workflow(self) -> StateGraph:
        """
        Create and configure the news-to-LinkedIn post agent workflow.

        Returns:
            StateGraph: The compiled workflow graph for the agent.
        """
        workflow = StateGraph(PostAgentState)
        
        # Add nodes
        workflow.add_node("search", self._search_news)
        workflow.add_node("analyze", self._analyze_news)
        workflow.add_node("select_news", self._select_news)
        workflow.add_node("generate_post", self._generate_post)
        # workflow.add_node("post_to_linkedin", self._post_to_linkedin)
        
        # Add edges
        workflow.add_edge("search", "analyze")
        workflow.add_edge("analyze", "select_news")
        workflow.add_edge("select_news", "generate_post")
        # workflow.add_edge("generate_post", "post_to_linkedin")
        workflow.add_edge("generate_post", END)
        
        # Set entry point
        workflow.set_entry_point("search")
        
        return workflow.compile()

    def run(self, topic: str) -> PostAgentState:
        """
        Run the LinkedIn post agent workflow with the given topic.

        Args:
            topic (str): The topic to search news and generate a LinkedIn post for.

        Returns:
            PostAgentState: The final state after running the workflow.
        """
        initial_state = {
            "messages": [],
            "topic": topic,
            "next_step": "search"
        }
        result = self.workflow.invoke(initial_state)
        return result