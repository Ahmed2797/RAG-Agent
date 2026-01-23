from crewai import Agent, Crew, Task, LLM
from crewai_tools import PDFSearchTool
# from ragagent.model import hf_llm
# from ragagent.tools import create_pdf_tool

# llm = hf_llm()
# pdf_tool = create_pdf_tool()


def pdf_agent(llm,pdf_tool):
    agent = Agent(
        role="PDF Knowledge Expert",
        goal="Answer questions based on PDF documents.",
        backstory="An AI assistant that extracts information from PDFs.",
        llm=llm,
        tools=[pdf_tool],  # tools হিসেবে দিন, knowledge_sources নয়
        # verbose=True
    )


    # -------------------------
    # Task
    # -------------------------
    pdf_task = Task(
        description=(
            "Answer the user query using ONLY the PDF content.\n"
            "Rules:\n"
            "1. Do NOT use prior knowledge\n"
            "2. Do NOT guess\n"
            "3. If the PDF does not contain the answer, respond with: NOT_FOUND\n\n"
            "User Question: {input}"
        ),
        agent=agent,
        expected_output="Answer from PDF or NOT_FOUND"
    )


    # -------------------------
    # Crew
    # -------------------------
    pdf_crew = Crew(
        agents=[agent],
        tasks=[pdf_task],
        # verbose=True
    )

    return pdf_crew


def wiki_agent(llm,wikipedia_search):
    # -------------------------
    # Wikipedia Agent
    # -------------------------
    wikipedia_agent = Agent(
        llm=llm,
        role="Wikipedia Research Agent",
        goal="Answer user questions using Wikipedia if PDF agent did not find an answer",
        backstory=(
            "You are an expert in summarizing Wikipedia content. "
            "Always give short, factual answers. "
            "If the topic is not found, respond with exactly: NOT_FOUND"
        ),
        tools=[wikipedia_search],
        # verbose=True,
    )

    # -------------------------
    # Task
    # -------------------------
    wikipedia_task = Task(
        description=(
            "Answer the user query using Wikipedia content.\n"
            "Rules:\n"
            "1. Search Wikipedia using the wikipedia_search tool.\n"
            "2. Provide concise and factual summary.\n"
            "3. If the topic is not found, respond with: NOT_FOUND\n"
            "User Question: {input}"
        ),
        agent=wikipedia_agent,
        expected_output="Answer from Wikipedia or NOT_FOUND"
    )

    # -------------------------
    # Crew
    # -------------------------
    wikipedia_crew = Crew(
        agents=[wikipedia_agent],
        tasks=[wikipedia_task],
        # verbose=True
    )

    return wikipedia_crew

