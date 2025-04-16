from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

_ANALAYZE_NEWS_PROMPT = (
    "Analyze this news article and extract key points, trends, and insights that would be relevant for a professional LinkedIn audience."
)
_GENERATE_POST_PROMPT = (
    "Based on the news analysis, write a LinkedIn post as if you're personally sharing your thoughts, takeaways, or reflections on the given topic title — not just reporting the news."
    "The tone should feel human, conversational, friendly and relatable."
    "You can express curiosity, surprise, excitement, or even questions you’re pondering, as long as it's authentic and adds professional value."
    "Do not start with Okay folks..., just start with your thoughts and insights."
    "Use light emojis if appropriate, include relevant hashtags, and wrap the post content only in pure text format with [START_POST] and [END_POST] markers."
)
_RANK_NEWS_PROMPT = (
    "Rank the following news articles based on their relevance to the topic and and your analysis. Return only the index number of the most relevant article (0-based)."
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
