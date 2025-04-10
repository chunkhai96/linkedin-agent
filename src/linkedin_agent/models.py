from langchain_google_genai import ChatGoogleGenerativeAI

# Initialize the LLM
def get_llm(stream: bool = False):
    """Initialize and return the Google Generative AI model.
    
    Args:
        stream: Whether to stream the response or not. Defaults to False.
    """
    return ChatGoogleGenerativeAI(
        model="gemini-2.0-flash",
        stream=stream
    )