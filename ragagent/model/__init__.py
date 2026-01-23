from crewai import LLM
from dotenv import load_dotenv
from openai import OpenAI
import os
load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
HUGGINGFACEHUB_API_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")

def groq_llm():
    groq_llm = LLM(
        model="llama-3.1-8b-instant",
        api_key= GROQ_API_KEY,temperature=0.7
    )

    return groq_llm

def hf_llm():
    llm = LLM(
        model = "huggingface/meta-llama/Llama-3.3-70B-Instruct",
        # model="huggingface/meta-llama/Llama-3.1-8B-Instruct",
        api_key= HUGGINGFACEHUB_API_TOKEN
    ) 

    return llm

def ask_openai(query: str, max_tokens: int = 500) -> str:
    """
    Ask OpenAI using system and user roles.

    Args:
        query (str): User input text
        max_tokens (int): Maximum output tokens

    Returns:
        str: Model response text
    """

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY not set")

    client = OpenAI(api_key=api_key)
    # messages = [
    #     {
    #         "role": "system",
    #         "content": "You are a job recommendation assistant."
    #     },
    #     {
    #         "role": "user",
    #         "content": query
    #     }
    # ]

    response = client.responses.create(
        model="gpt-4o-mini",          # cheapest good-performance model
        input=query,
        max_output_tokens=max_tokens
    )

    return response.output_text

