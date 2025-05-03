# LinkedIn Agent

## Project Description
This project provides an automated LinkedIn posting agent that creates and publishes posts based on news topics. It utilizes LangChain and LangGraph to create a workflow that searches news, analyzes content, generates posts, and publishes them to LinkedIn.

## Installation
```bash
poetry install
```

## Usage
```python
from linkedin_agent.agents.post_agent import LinkedInPostAgent

agent = LinkedInPostAgent(stream=stream)
result = agent.run("YOUR TOPIC HERE")
```

## Configuration

1. Copy the example environment file:
```bash
cp .env.example .env
```
2. Add your tokens to the .env file:
```
LINKEDIN_ACCESS_TOKEN = 'YOUR_TOKEN_HERE'
GOOGLE_API_KEY = 'YOUR_TOKEN_HERE'
SUPABASE_URL = 'YOUR_SUPABASE_URL_HERE'
SUPABASE_KEY = 'YOUR_SUPABASE_KEY_HERE'
```