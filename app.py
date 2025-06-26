import os
import logging
from fastapi import FastAPI, Query, HTTPException, Path
from typing import List, TypedDict, Optional
from dotenv import load_dotenv
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from datetime import datetime

# --- LangChain & Google Imports ---
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.retrievers import BaseRetriever

# --- Database Imports ---
from sqlalchemy import select

from db import SessionLocal, Document, init_db, ChatSession, Message, User
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
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)
    logging.info("Google AI Models initialized successfully.")
except Exception as e:
    logging.error(f"Failed to initialize Google AI Models: {e}")
    raise

# --- Database Retriever ---
class SQLAlchemyRetriever(BaseRetriever):
    """
    A LangChain-compatible retriever that fetches documents from a PostgreSQL/PGVector database.
    """
    def invoke(self, input: str, **kwargs) -> List[Document]:
        logging.info(f"Retrieving documents for query: '{input}'")
        query_embedding = embeddings_model.embed_query(input)

        with SessionLocal() as db:
            stmt = select(Document).order_by(Document.embedding.l2_distance(query_embedding)).limit(3)
            results = db.execute(stmt).scalars().all()
            logging.info(f"Found {len(results)} relevant documents.")
            return results

    def _get_relevant_documents(self, query: str, *, run_manager=None):
        # Dummy implementation to satisfy BaseRetriever's abstract method
        return []

retriever = SQLAlchemyRetriever()

# --- LangGraph State and Nodes ---
class GraphState(TypedDict):
    question: str
    context: str
    answer: str

def retrieve_node(state: GraphState):
    logging.info("Node: retrieve")
    question = state["question"]
    docs = retriever.invoke(question)
    context = "\n\n---\n\n".join([str(doc.content) for doc in docs])
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
def query_rag(
    q: str = Query(..., description="The question you want to ask"),
    user_id: int = Query(..., description="The user ID for the chat")
):
    inputs = {"question": q}
    result = rag_app.invoke(inputs)
    # Store the chat session and message in the database, linked to the user
    with SessionLocal() as db:
        session = ChatSession(user_id=user_id)
        db.add(session)
        db.flush()  # To get session.id
        message = Message(
            session_id=session.id,
            question=q,
            context=result.get("context"),
            answer=result.get("answer")
        )
        db.add(message)
        db.commit()
    return {"answer": result["answer"], "context": result["context"]}

# --- Pydantic Schemas for User CRUD ---
class UserCreate(BaseModel):
    username: str
    email: Optional[str] = None

class UserRead(BaseModel):
    id: int
    username: str
    email: Optional[str] = None
    created_at: datetime
    class Config:
        from_attributes = True

class UserUpdate(BaseModel):
    username: Optional[str] = None
    email: Optional[str] = None

# --- User CRUD Endpoints ---
@app.post("/users/", response_model=UserRead)
def create_user(user: UserCreate):
    with SessionLocal() as db:
        db_user = db.query(User).filter(User.username == user.username).first()
        if db_user:
            raise HTTPException(status_code=400, detail="Username already registered")
        new_user = User(username=user.username, email=user.email)
        db.add(new_user)
        db.commit()
        db.refresh(new_user)
        return new_user

@app.get("/users/", response_model=List[UserRead])
def read_users():
    with SessionLocal() as db:
        users = db.query(User).all()
        return users

@app.get("/users/{user_id}", response_model=UserRead)
def read_user(user_id: int = Path(..., description="The ID of the user to retrieve")):
    with SessionLocal() as db:
        user = db.query(User).filter(User.id == user_id).first()
        if not user:
            raise HTTPException(status_code=404, detail="User not found")
        return user

@app.put("/users/{user_id}", response_model=UserRead)
def update_user(user_id: int, user_update: UserUpdate):
    with SessionLocal() as db:
        user = db.query(User).filter(User.id == user_id).first()
        if not user:
            raise HTTPException(status_code=404, detail="User not found")
        if user_update.username is not None:
            user.username = user_update.username
        if user_update.email is not None:
            user.email = user_update.email
        db.commit()
        db.refresh(user)
        return user

@app.delete("/users/{user_id}")
def delete_user(user_id: int):
    with SessionLocal() as db:
        user = db.query(User).filter(User.id == user_id).first()
        if not user:
            raise HTTPException(status_code=404, detail="User not found")
        db.delete(user)
        db.commit()
        return {"detail": "User deleted"}