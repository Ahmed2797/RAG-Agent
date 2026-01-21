# RAG-Agent

## PDF Chat Assistant

A **fast, production-ready PDF question-answering assistant** built with **CrewAI** and **FastAPI**, using an **agentic RAG architecture**:

- **Primary source**: PDF documents (strict PDF-grounded answers)
- **Fallback**: Wikipedia (if PDF does not contain the answer)
- **Caching**: Embeddings are created **once per PDF**, ensuring fast responses
- **Multi-PDF support**: Upload, switch, and delete PDFs dynamically

---

## 🚀 Features

- Upload and manage multiple PDFs
- Answer user queries strictly from PDF content
- Wikipedia fallback if PDF does not contain relevant info
- Deterministic, source-tagged answers
- Fast performance with embedding caching
- REST API for easy frontend integration
- Health check and active PDF tracking
- Production-ready FastAPI backend

---

## 🏗 Architecture

    mermaid
    flowchart LR
    A[Browser / Frontend] --> B[FastAPI /api/chat]
    B --> C[Pipeline(query, pdf_path)]
    C --> D[PDF Agent]
    D -->|NOT_FOUND| E[Wikipedia Agent]
    D -->|Answer Found| F[Final Answer + Source]
    E --> F
