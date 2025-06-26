import os
import time
import logging
from db import SessionLocal, Document, init_db
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter

# --- Setup ---
load_dotenv()
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Constants ---
# CORRECT model for text embeddings
EMBED_MODEL = "models/embedding-001"
# The API allows 60 requests per minute. We'll be safe and do 1 every 1.1 seconds.
REQUESTS_PER_MINUTE = 60
DELAY_BETWEEN_REQUESTS = 60.0 / REQUESTS_PER_MINUTE

# --- Core Functions ---

def initialize_embeddings() -> GoogleGenerativeAIEmbeddings:
    """Initializes and returns the Google Generative AI embeddings model."""
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        logging.error("GOOGLE_API_KEY not found in environment variables.")
        raise ValueError("Please set the GOOGLE_API_KEY in your .env file.")
    
    logging.info(f"Initializing embeddings with model: {EMBED_MODEL}")
    return GoogleGenerativeAIEmbeddings(model=EMBED_MODEL, google_api_key=api_key)

def ingest_document(file_path: str, embeddings_model: GoogleGenerativeAIEmbeddings):
    """
    Reads a document, splits it into chunks, generates embeddings for each chunk,
    and stores them in the database, respecting API rate limits.
    """
    try:
        logging.info(f"Reading content from '{file_path}'...")
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        if not content.strip():
            logging.warning(f"Document '{file_path}' is empty. Skipping.")
            return

        # 1. Split the document into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,  # The size of each chunk in characters
            chunk_overlap=100 # The overlap between chunks
        )
        chunks = text_splitter.split_text(content)
        
        logging.info(f"Document split into {len(chunks)} chunks.")

        # 2. Process each chunk with a delay
        with SessionLocal() as db:
            for i, chunk in enumerate(chunks):
                logging.info(f"Processing chunk {i + 1}/{len(chunks)}...")
                
                # Generate embedding for the chunk
                vector = embeddings_model.embed_query(chunk)
                
                # Store the chunk and its embedding
                doc = Document(content=chunk, embedding=vector)
                db.add(doc)
                
                logging.info(f"Chunk {i + 1} embedded and added to session.")

                # IMPORTANT: Wait before the next request to avoid hitting the rate limit
                time.sleep(DELAY_BETWEEN_REQUESTS) 
            
            # 3. Commit all chunks to the database at once
            logging.info("Committing all chunks to the database...")
            db.commit()

        logging.info(f"Successfully embedded and stored all {len(chunks)} chunks.")

    except FileNotFoundError:
        logging.error(f"Error: The file '{file_path}' was not found.")
    except Exception as e:
        logging.error(f"An unexpected error occurred: {e}", exc_info=True)

# --- Main Execution ---

def main():
    """Main function to run the ingestion process."""
    try:
        init_db()
        embeddings = initialize_embeddings()
        document_to_ingest = "doc.txt"
        ingest_document(document_to_ingest, embeddings)
    except ValueError as e:
        logging.error(f"A configuration error occurred: {e}")
    except Exception as e:
        logging.error(f"A critical error stopped the main process: {e}")

if __name__ == "__main__":
    main()