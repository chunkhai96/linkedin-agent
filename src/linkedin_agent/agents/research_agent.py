import requests
import json
import os
import re
import time
import sqlite3
from langgraph.graph import StateGraph, END
from datetime import datetime, timedelta
from typing import List, Dict, Any
from bs4 import BeautifulSoup
from supabase import create_client, Client
from tqdm import tqdm

from .post_agent import LinkedInPostAgent
from ..types import ResearchAgentState, PostingPlan, PostAgentState
from ..models import get_llm
from ..prompts import analyze_news_prompt, generate_post_prompt, rank_news_prompt, select_news_prompt, plan_post_prompt

class LinkedInResearchAgent:
    def __init__(self, stream: bool = False, topics: List[str] = None):
        """Initialize the LinkedIn Research Agent with configurable streaming and topics."""
        self.llm = get_llm(stream=stream)
        self.workflow = self._create_workflow()
        self.topics = topics or ["artificial-intelligence"]
        self.database_client = self._connect_database()

    def _connect_database(self) -> Client:
        """Connect to supabase database.
        
        Returns:
            Client: A configured Supabase client instance
        """
        return create_client(os.getenv("SUPABASE_URL"), os.getenv("SUPABASE_KEY"))

    def _scrape_techcrunch(self, state: ResearchAgentState) -> ResearchAgentState:
        """Scrape TechCrunch for articles on specified topics within the last 7 days.
        
        Args:
            state: The current ResearchAgentState containing workflow messages
            
        Returns:
            ResearchAgentState: Updated state with scraped articles in news_content
        """
        print(f"[STEP 1/5] Scraping TechCrunch for articles on topics: {self.topics}")
        
        # Calculate date 7 days ago
        seven_days_ago = datetime.now() - timedelta(days=7)
        
        all_articles = []
        
        # Scrape TechCrunch homepage and topic pages
        urls = ["https://techcrunch.com/"] + [f"https://techcrunch.com/category/{topic.lower().replace(' ', '-')}/" 
                                            for topic in self.topics]
        
        for url in urls:
            try:
                response = requests.get(url, headers={"User-Agent": "Mozilla/5.0"})
                soup = BeautifulSoup(response.content, 'html.parser')
                
                # Also find the new loop-card title format
                loop_card_titles = soup.find_all('h3', class_='loop-card__title')
                
                # Process articles from loop-card format
                for title_elem in loop_card_titles:
                    # Extract title and link from the loop-card title
                    link_elem = title_elem.find('a')
                    if not link_elem:
                        continue
                        
                    title = link_elem.get_text().strip()
                    link = link_elem['href'] if 'href' in link_elem.attrs else None
                    
                    # Skip if no link or title
                    if not link or not title:
                        continue
                    
                    # For loop-card format, we might not have date or snippet
                    # Add with empty snippet since we can't easily find related content
                    all_articles.append({
                        "title": title,
                        "link": link,
                        "snippet": "",
                        "source": "TechCrunch"
                    })
            except Exception as e:
                print(f"Error scraping {url}: {str(e)}")
        
        # Convert list to JSON text
        articles_json = json.dumps(all_articles, indent=2)
        print(f"Found {len(all_articles)} articles")
        
        return {
            "messages": state["messages"],
            "news_content": {"raw_content": articles_json},
            "next_step": "select_news"
        }

    def _select_news(self, state: ResearchAgentState) -> ResearchAgentState:
        """Select the most relevant news items based on the analysis.
        
        Args:
            state: The current ResearchAgentState containing scraped articles
            
        Returns:
            ResearchAgentState: Updated state with selected news items
        """
        print(f"[STEP 2/5] Selecting most relevant news items")
        news_items = state["news_content"]["raw_content"]
        
        messages = select_news_prompt.format(messages=[{
            "role": "human",
            "content": "News Articles:\n{}".format(state["news_content"]["raw_content"])
        }])
        response = self.llm.invoke(messages)
        
        try:
            # Clean markdown formatting before parsing
            content = response.content.strip()
            json_match = re.search(r'```json\n(.*?)\n```', content, re.DOTALL)
            if json_match:
                content = json_match.group(1).strip()
            # Parse the response to get selected indices
            selected_indices = json.loads(content)
            news_items_list = json.loads(news_items)
            
            # Get the selected news items
            selected_news_items = [news_items_list[idx] for idx in selected_indices if idx < len(news_items_list)]
        except (ValueError, json.JSONDecodeError, IndexError):
            # Fallback to first 7 items if parsing fails
            news_items_list = json.loads(news_items)
            selected_news_items = news_items_list[:min(7, len(news_items_list))]
        
        return {
            "messages": [*state["messages"], *messages, response],
            "news_content": state["news_content"],
            "selected_news": selected_news_items,
            "next_step": "plan_posts"
        }

    def _plan_posts(self, state: ResearchAgentState) -> ResearchAgentState:
        """Plan LinkedIn posts for the week based on selected news items.
        
        Args:
            state: The current ResearchAgentState containing selected news
            
        Returns:
            ResearchAgentState: Updated state with posting schedule
        """
        print(f"[STEP 3/5] Planning LinkedIn posts for the week")

        selected_news = state["selected_news"]
        
        messages = plan_post_prompt.format(messages=[{
            "role": "human",
            "content": "Selected News Articles:\n{selected_news}".format(selected_news=json.dumps(selected_news))
        }])
        response = self.llm.invoke(messages)
        
        try:
            # Clean markdown formatting before parsing
            content = response.content.strip()
            json_match = re.search(r'```json\n(.*?)\n```', content, re.DOTALL)
            if json_match:
                content = json_match.group(1).strip()
            
            # Parse the response to get the posting plan
            posting_plans = json.loads(content)
            for plan in posting_plans:
                plan["title"] = selected_news[plan["article_index"]]["title"]
                plan["link"] = selected_news[plan["article_index"]]["link"]
                plan["snippet"] = selected_news[plan["article_index"]]["snippet"]
        except (ValueError, json.JSONDecodeError):
            # Fallback to a simple plan if parsing fails
            posting_plans: List[PostingPlan] = []
            days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
            article_index = 0
            
            for day in days:
                # Add 1-2 posts per day
                for _ in range(min(2, len(selected_news) - article_index)):
                    if article_index < len(selected_news):
                        posting_plans.append({
                            "day": day,
                            "article_index": article_index,
                            "title": selected_news[article_index]["title"],
                            "link": selected_news[article_index]["link"],
                            "snippet": selected_news[article_index]["snippet"],
                            "reason": f"Relevant article for {day}"
                        })
                        article_index += 1
        
        return {
            "messages": [*state["messages"], *messages, response],
            "news_content": state["news_content"],
            "selected_news": selected_news,
            "posting_plans": posting_plans,
            "next_step": "save_to_database"
        }

    def _write_post(self, state: ResearchAgentState) -> ResearchAgentState:
        """Generate LinkedIn posts based on the selected news items.
        
        Args:
            state: The current ResearchAgentState containing posting schedule
            
        Returns:
            ResearchAgentState: Updated state with generated post content
        """
        print(f"[STEP 4/5] Generating LinkedIn posts")
        
        post_agent = LinkedInPostAgent(stream=False, verbose=0)
        posting_plans = state["posting_plans"]
        
        for plan in tqdm(posting_plans):
            topic = plan["title"]
            post_agent_state: PostAgentState = {
                "messages": [],
                "topic": topic,
                "next_step": "search"
            }
            
            max_retries = 3
            base_delay = 60
            
            for attempt in range(max_retries):
                try:
                    post_result = post_agent.run(post_agent_state)
                    plan["post_content"] = post_result["linkedin_post"]
                    break
                except Exception as e:
                    if "quota" in str(e).lower() and attempt < max_retries - 1:
                        print(f"Rate limit hit, retrying in {base_delay} seconds... (Attempt {attempt + 1}/{max_retries})")
                        time.sleep(base_delay)
                    else:
                        print(f"Failed to generate post for '{topic}': {str(e)}")
                        plan["post_content"] = None
                        break
            
        return {
            "messages": state["messages"],
            "news_content": state["news_content"],
            "selected_news": state["selected_news"],
            "posting_plans": posting_plans,
            "next_step": "save_to_database"
        }
        
    def _save_to_database(self, state: ResearchAgentState) -> ResearchAgentState:
        """Save the generated posts to the SQLite database.
        
        Args:
            state: The current ResearchAgentState containing posts to save
            
        Returns:
            ResearchAgentState: Final state after saving to database
        """
        print(f"[STEP 5/5] Saving posts to database")

        try:
            response = (
                self.database_client.table("article")
                .update({"expired": True})
                .is_("expired", None)
                .execute()
            )
            print(f"Set old articles to expired")
        except Exception as exception:
            print(f"Error in setting old articles to expired: {str(exception)}")
        
        posting_plans = state["posting_plans"]
        
        # Add timestamp to each post and save to database
        articles_to_save = []
        for plan in posting_plans:
            articles_to_save.append({
                "title": plan["title"],
                "link": plan["link"],
                "day": plan["day"],
                "reason": plan["reason"],
                "snippet": plan["snippet"],
                "post_content": plan["post_content"],
            })

        try:
            response = (
                self.database_client.table("article")
                .insert(articles_to_save)
                .execute()
            )
            print(f"Successfully saved {len(articles_to_save)} articles to database")
        except Exception as exception:
            print(f"Error saving articles to database: {str(exception)}")

        return {
            "messages": state["messages"],
            "news_content": state["news_content"],
            "selected_news": state["selected_news"],
            "posting_plans": state["posting_plans"],
            "next_step": END
        }

    def _create_workflow(self) -> StateGraph:
        """Create and configure the TechCrunch-to-LinkedIn research agent workflow."""
        workflow = StateGraph(ResearchAgentState)
        
        # Add nodes
        workflow.add_node("scrape", self._scrape_techcrunch)
        workflow.add_node("select_news", self._select_news)
        workflow.add_node("plan_posts", self._plan_posts)
        workflow.add_node("write_post", self._write_post)
        workflow.add_node("save_to_database", self._save_to_database)
        
        # Add edges
        workflow.add_edge("scrape", "select_news")
        workflow.add_edge("select_news", "plan_posts")
        workflow.add_edge("plan_posts", "write_post")
        workflow.add_edge("write_post", "save_to_database")
        workflow.add_edge("save_to_database", END)
        
        # Set entry point
        workflow.set_entry_point("scrape")
        
        return workflow.compile()

    def run(self, topics: List[str] = None) -> ResearchAgentState:
        """Run the LinkedIn research agent workflow with optional topics override.
        
        Args:
            topics: Optional list of topics to override the default topics
            
        Returns:
            ResearchAgentState: The final state after running the workflow
        """
        if topics:
            self.topics = topics
            
        initial_state = {
            "messages": [],
            "next_step": "scrape"
        }
        return self.workflow.invoke(initial_state)