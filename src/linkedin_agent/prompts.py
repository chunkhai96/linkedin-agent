from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

_RESEARCHER_SYSTEM_PROMPT = (
    "You are a world class LinkedIn post researcher with a passion for sharing valuable insights and staying updated with the latest industry trends. "
    "Your specialty is researching and curating high-quality tech/AI/business content. "
    "The focus areas are Artificial Intelligence/Machine Learning, Software Engineering, Tech Business Trends, Startup Ecosystem, and Emerging Technologies. "
    "The goal is to select the most relevant, engaging and valuable topics. Then create a detail plan for the week's posts. "
)
_ANALAYZE_NEWS_PROMPT = (
    "Analyze this news article and extract key points, trends, and insights that would be relevant for a professional LinkedIn audience."
)
_GENERATE_POST_PROMPT = (
    "Based on the news analysis, write a LinkedIn post as if you're personally sharing your thoughts, takeaways, or reflections on the given topic title — not just reporting the news. "
    "The tone should feel human, conversational, friendly and relatable. "
    "You can express curiosity, surprise, excitement, or even questions you’re pondering, as long as it's authentic and adds professional value. "
    "Do not start with Okay folks..., just start with your thoughts and insights. "
    "Use light emojis if appropriate, include relevant hashtags, and wrap the post content only in pure text format with [START_POST] and [END_POST] markers. "
)
_RANK_NEWS_PROMPT = (
    "Rank the following news articles based on their relevance to the topic and and your analysis. Return only the index number of the most relevant article (0-based)."
)
_SELECT_NEWS_PROMPT = (
    "Rank the following news articles based on their relevance and importance. "
    "Select the top 7-14 articles that would make good LinkedIn posts (1-2 posts per day for a week). "
    "Avoid the previously posted topics. "
    "Return a JSON array of indices (0-based) for the selected articles."
)
_PLAN_POST_PROMPT = (
    "Create a posting schedule for LinkedIn for the next 7 days using these {len(selected_news)} articles. "
    "Plan 1-2 posts per day, with a good mix of topics throughout the week. "
    "For each post, include: "
    "1. The day of the week "
    "2. The article index from the provided list "
    "3. A brief reason why this article is worth posting about "
    "Return the plan as a JSON array with objects containing 'day', 'article_index', and'reason' fields."
)

# Create prompt templates for different steps
analyze_news_prompt = ChatPromptTemplate.from_messages([
    MessagesPlaceholder(variable_name="messages"),
    ("human", _ANALAYZE_NEWS_PROMPT)
])

generate_post_prompt = ChatPromptTemplate.from_messages([
    MessagesPlaceholder(variable_name="messages"),
    ("human", _GENERATE_POST_PROMPT)
])

rank_news_prompt = ChatPromptTemplate.from_messages([
    MessagesPlaceholder(variable_name="messages"),
    ("human", _RANK_NEWS_PROMPT)
])

select_news_prompt = ChatPromptTemplate.from_messages([
    ("human", _RESEARCHER_SYSTEM_PROMPT),
    MessagesPlaceholder(variable_name="messages"),
    ("human", _SELECT_NEWS_PROMPT)
])
plan_post_prompt = ChatPromptTemplate.from_messages([
    ("human", _RESEARCHER_SYSTEM_PROMPT),
    MessagesPlaceholder(variable_name="messages"),
    ("human", _PLAN_POST_PROMPT)
])