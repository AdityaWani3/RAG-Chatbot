from llama_index.core import VectorStoreIndex
from llama_index.embeddings.gemini import GeminiEmbedding  # Ensure this is the correct import for Gemini embedding
from llama_index.llms.openai import OpenAI  # If you're using OpenAI models, adjust if Gemini is used
from llama_index.core import Settings  # Settings replaces ServiceContext
from llama_index.core.node_parser import SentenceSplitter

from RAGchatbot.data_ingestion import load_data
from RAGchatbot.model_api import load_model

import sys
from exception import customexception
from logger import logging

def download_gemini_embedding(model, document):
    """
    Downloads and initializes a Gemini Embedding model for vector embeddings.

    Returns:
    - VectorStoreIndex: An index of vector embeddings for efficient similarity queries.
    """
    try:
        logging.info("Initializing Gemini embedding model...")
        
        # Set up Gemini embedding model in Settings
        gemini_embed_model = GeminiEmbedding(model_name="models/embedding-001")
        
        # Set up the models and settings
        Settings.llm = model  # Set the LLM model (replace `model` with the correct Gemini LLM if available)
        Settings.embed_model = gemini_embed_model  # Use Gemini embedding model
        Settings.node_parser = SentenceSplitter(chunk_size=1000, chunk_overlap=50)
        Settings.num_output = 512
        Settings.context_window = 3900
        
        logging.info("Creating index from documents...")
        
        # Create an index from documents
        index = VectorStoreIndex.from_documents(document, service_context=Settings)
        
        # Persist the storage context (save index)
        index.storage_context.persist()
        
        logging.info("Index created and persisted successfully.")
        
        # Create the query engine from the index
        query_engine = index.as_query_engine()
        return query_engine
        
    except Exception as e:
        # Log the error and raise a custom exception
        raise customexception(e, sys)
