from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

# Create prompt templates for different steps
analyze_news_prompt = ChatPromptTemplate.from_messages([
    MessagesPlaceholder(variable_name="messages"),
    ("human", "Analyze this news article and extract key points, trends, and insights that would be relevant for a professional LinkedIn audience.")
])

generate_post_prompt = ChatPromptTemplate.from_messages([
    MessagesPlaceholder(variable_name="messages"),
    ("human", "Based on the news analysis, create an engaging LinkedIn post that highlights the key insights and adds professional value. Include relevant hashtags and maintain a professional tone. Wrap your post content only in pure text format with [START_POST] and [END_POST] markers.")
])