import os
import logging
from fastapi import FastAPI, Query
from typing import List, TypedDict
from dotenv import load_dotenv
from fastapi.middleware.cors import CORSMiddleware

# --- LangChain & Google Imports ---
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.retrievers import BaseRetriever

# --- Database Imports ---
from sqlalchemy import select

from db import SessionLocal, Document, init_db
# ----------------- CHANGE 1: REMOVE THE UNUSED IMPORT -----------------
# No longer needed, as we will call the method directly on the column.
# from pgvector.sqlalchemy import vector_l2_distance

# --- LangGraph and LangSmith Imports ---
from langgraph.graph import StateGraph, END
from langsmith import Client

# --- Setup ---
load_dotenv()
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Configuration & Clients ---
os.environ["LANGCHAIN_TRACING_V2"] = "true" 
langsmith_client = Client()

# --- Models ---
try:
    embeddings_model = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0)
    logging.info("Google AI Models initialized successfully.")
except Exception as e:
    logging.error(f"Failed to initialize Google AI Models: {e}")
    raise

# --- Database Retriever ---
class SQLAlchemyRetriever(BaseRetriever):
    """
    A LangChain-compatible retriever that fetches documents from a PostgreSQL/PGVector database.
    """
    def _get_relevant_documents(self, query: str, *, run_manager) -> List[Document]:
        logging.info(f"Retrieving documents for query: '{query}'")
        query_embedding = embeddings_model.embed_query(query)

        with SessionLocal() as db:
            # ------------- CHANGE 2: UPDATE THE QUERY SYNTAX -------------
            # Use the .l2_distance() method directly on the embedding column
            stmt = select(Document).order_by(Document.embedding.l2_distance(query_embedding)).limit(3)
            # -------------------------------------------------------------
            results = db.execute(stmt).scalars().all()
            logging.info(f"Found {len(results)} relevant documents.")
            return results

retriever = SQLAlchemyRetriever()

# --- LangGraph State and Nodes ---
class GraphState(TypedDict):
    question: str
    context: str
    answer: str

def retrieve_node(state: GraphState):
    logging.info("Node: retrieve")
    question = state["question"]
    docs = retriever.get_relevant_documents(question)
    context = "\n\n---\n\n".join([doc.content for doc in docs])
    return {"context": context, "question": question}

def generate_node(state: GraphState):
    logging.info("Node: generate")
    context = state["context"]
    question = state["question"]
    
    prompt_template = """You are an assistant for question-answering tasks. 
Use the following pieces of retrieved context to answer the question. 
If you don't know the answer, just say that you don't know. 
Be concise.

Context:
{context}

Question:
{question}

Answer:"""
    prompt = ChatPromptTemplate.from_template(prompt_template)
    
    rag_chain = (
        prompt
        | llm
        | StrOutputParser()
    )
    
    answer = rag_chain.invoke({"context": context, "question": question})
    return {"answer": answer}

# --- Build the Graph ---
workflow = StateGraph(GraphState)
workflow.add_node("retrieve", retrieve_node)
workflow.add_node("generate", generate_node)
workflow.set_entry_point("retrieve")
workflow.add_edge("retrieve", "generate")
workflow.add_edge("generate", END)
rag_app = workflow.compile()
logging.info("LangGraph RAG app compiled successfully.")

# --- FastAPI Application ---
app = FastAPI(
    title="Agentic RAG with Gemini and LangGraph",
    description="A FastAPI application for querying documents using a RAG agent.",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
def startup_event():
    init_db()

@app.get("/query", summary="Ask a question to the RAG agent")
def query_rag(q: str = Query(..., description="The question you want to ask")):
    inputs = {"question": q}
    result = rag_app.invoke(inputs)
    return {"answer": result["answer"], "context": result["context"]}