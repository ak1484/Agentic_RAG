# Agentic RAG App

This project is a Retrieval-Augmented Generation (RAG) application using:
- **PostgreSQL** with `pgvector` for vector storage
- **LangGraph** for agentic workflows
- **LangSmith** for tracing
- **Gemini LLM** (Google Generative AI) for structured answers and embeddings

## Features
- Ingest and embed documents (e.g., `doc.txt`)
- Store and search embeddings in PostgreSQL with pgvector
- Agentic retrieval and answer generation
- Tracing with LangSmith

## Setup

### 1. Clone the repository
```bash
# Clone this repo and cd into it
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Set up PostgreSQL with pgvector
- Install PostgreSQL (v13+ recommended)
- Install the `pgvector` extension:
  ```sql
  CREATE EXTENSION IF NOT EXISTS vector;
  ```
- Create a database for the app.

### 4. Environment Variables
Create a `.env` file with the following:
```
DATABASE_URL=postgresql+psycopg://user:password@localhost:5432/yourdb
gemini_api_key=YOUR_GEMINI_API_KEY
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=YOUR_LANGSMITH_API_KEY
LANGCHAIN_PROJECT=agentic-rag
```

### 5. Ingest Example Document
Place your `doc.txt` in the project root. Run the embedding script (to be provided) to ingest and embed the document.

### 6. Run the App
```bash
uvicorn app:app --reload
```

## Usage
- Use the API or CLI to query the RAG system.

## TODO
- [ ] Add embedding and ingestion script
- [ ] Implement RAG pipeline
- [ ] Add API endpoints 