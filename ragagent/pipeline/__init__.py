# ragagent/pipeline.py

from crewai import Agent, Crew, Task, LLM
from crewai_tools import PDFSearchTool
from crewai.tools import tool
from typing import Dict
import os
import wikipedia
from dotenv import load_dotenv
from openai import OpenAI
import os
load_dotenv()

# --------------------------
# LLM
# --------------------------

llm = LLM(
    model="openai/gpt-4o-mini",
    api_key=os.getenv("OPENAI_API_KEY"),
    temperature=0.3,
    max_tokens=500
)

# --------------------------
# CACHE
# --------------------------
PDF_CREW_CACHE: Dict[str, Crew] = {}

# --------------------------
# PDF TOOL
# --------------------------
def create_pdf_tool(pdf_path: str) -> PDFSearchTool:
    return PDFSearchTool(
        pdf=pdf_path,
        config={
            "embedding_model": {
                "provider": "sentence-transformer",
                "config": {"model": "all-MiniLM-L6-v2"}
            },
            "vectordb": {"provider": "chromadb", "config": {}}
        }
    )

# --------------------------
# PDF AGENT (FORCED TOOL USAGE)
# --------------------------
def pdf_agent(llm, pdf_tool) -> Crew:
    agent = Agent(
        role="PDF Knowledge Expert",
        goal="Answer strictly from the PDF",
        backstory="You MUST search the PDF using the PDFSearchTool.",
        llm=llm,
        tools=[pdf_tool],
        verbose=True
    )

    task = Task(
        description=(
            "You MUST use the PDFSearchTool to answer.\n"
            "Steps:\n"
            "1. Search the PDF using PDFSearchTool\n"
            "2. Answer ONLY from search results\n"
            "3. If nothing relevant is found, reply EXACTLY: NOT_FOUND\n\n"
            "User Question: {input}"
        ),
        agent=agent,
        expected_output="PDF answer or NOT_FOUND"
    )

    return Crew(
        agents=[agent],
        tasks=[task],
        verbose=True
    )

# --------------------------
# WIKIPEDIA TOOL
# --------------------------
@tool("Wikipedia Search")
def wikipedia_search(query: str) -> str:
    """
    Searches Wikipedia and returns a short summary (up to 4 sentences).
    """
    try:
        return wikipedia.summary(query, sentences=4)
    except wikipedia.DisambiguationError as e:
        return wikipedia.summary(e.options[0], sentences=4)
    except Exception:
        return "NOT_FOUND"

# --------------------------
# WIKI AGENT
# --------------------------
def wiki_agent(llm) -> Crew:
    agent = Agent(
        role="Wikipedia Research Agent",
        goal="Answer from Wikipedia only",
        backstory="You are an expert researcher who searches Wikipedia and summarizes accurate information clearly.",
        llm=llm,
        tools=[wikipedia_search],
        verbose=True
    )

    task = Task(
        description=(
            "Search Wikipedia and answer.\n"
            "If nothing found, reply EXACTLY: NOT_FOUND\n\n"
            "User Question: {input}"
        ),
        agent=agent,
        expected_output="Wikipedia answer or NOT_FOUND"
    )

    return Crew(
        agents=[agent],
        tasks=[task],
        verbose=True
    )

# --------------------------
# GET PDF CREW (CACHED)
# --------------------------
def get_pdf_crew(pdf_path: str) -> Crew:
    if pdf_path in PDF_CREW_CACHE:
        return PDF_CREW_CACHE[pdf_path]

    crew = pdf_agent(llm, create_pdf_tool(pdf_path))
    PDF_CREW_CACHE[pdf_path] = crew
    return crew

# --------------------------
# PIPELINE
# --------------------------
def pipeline(query: str, pdf_path: str) -> str:
    pdf_crew = get_pdf_crew(pdf_path)
    pdf_result = pdf_crew.kickoff(inputs={"input": query})
    pdf_answer = pdf_result.raw

    if pdf_answer.upper() == "NOT_FOUND" or pdf_answer == "":
        wiki_crew = wiki_agent(llm)
        wiki_result = wiki_crew.kickoff(inputs={"input": query})
        wiki_answer = (wiki_result.raw or "NOT_FOUND").strip()
        return f"[SOURCE: WIKIPEDIA]\n{wiki_answer}"

    return f"[SOURCE: PDF]\n{pdf_answer}"
