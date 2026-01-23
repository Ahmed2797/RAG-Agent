from crewai_tools import PDFSearchTool

def create_pdf_tool(pdf_path):
    return PDFSearchTool(
        pdf=pdf_path,
        config={
            "embedding_model": {
                "provider": "sentence-transformer",
                "config": {"model": "all-MiniLM-L6-v2"}
            },
            "vectordb": {
                "provider": "chromadb",
                "config": {}
            }
        }
    )



from crewai.tools import tool
import wikipedia


@tool("Wikipedia Search")
def wikipedia_search(query: str) -> str:
    """
    Searches Wikipedia and returns a short summary (up to 4 sentences).
    """
    try:
        summary = wikipedia.summary(query, sentences=4)
        return summary
    except wikipedia.DisambiguationError as e:
        # If multiple options exist, pick first
        return wikipedia.summary(e.options[0], sentences=4)
    except Exception:
        return "NOT_FOUND"    



from crewai_tools import SerperDevTool

def serper_tool():
    serper_search = SerperDevTool()
    return serper_search