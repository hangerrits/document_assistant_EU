#!/usr/bin/env python3
"""
Document Set Q&A API

API for creating named sets of documents and performing Q&A using llama-index and ChromaDB.
"""

import os
import json
import uuid
import logging
import urllib.parse
import shutil
import time
import re # For sanitizing names
import asyncio
from typing import List, Dict, Any, Optional, Tuple
from contextlib import asynccontextmanager
from datetime import datetime
import tempfile
import glob # Make sure this is imported

from fastapi import FastAPI, HTTPException, Request, UploadFile, Form, File, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response, JSONResponse
from pydantic import BaseModel

from dotenv import load_dotenv

# Import llama-index components
from llama_index.core import VectorStoreIndex, StorageContext, load_index_from_storage, PromptTemplate
from llama_index.core.node_parser import SimpleNodeParser, NodeParser
from llama_index.core.storage.docstore import SimpleDocumentStore
from llama_index.core.storage.index_store import SimpleIndexStore
from llama_index.readers.docling import DoclingReader
from llama_index.core import Settings # Import Settings
# LLM Client imports - Use OpenAILike for compatibility with custom model names
from llama_index.llms.openai_like import OpenAILike
# from llama_index.llms.openai import OpenAI as LlamaOpenAI # <-- This has strict model name validation
# Embedding Client imports
from llama_index.core.base.embeddings.base import BaseEmbedding, Embedding
from openai import OpenAI as StandardOpenAI # Import standard client for custom class
# from llama_index.embeddings.huggingface_api import HuggingFaceInferenceAPIEmbedding # <-- Ignored base_url
# from llama_index.embeddings.openai import OpenAIEmbedding # <-- Didn't work with Nebius custom model name
#from llama_index.embeddings.adapter.base import AdapterEmbeddingModel # Try Adapter model
from llama_index.node_parser.docling import DoclingNodeParser
from llama_index.vector_stores.chroma import ChromaVectorStore
import chromadb
import numpy as np
from docling.chunking import HybridChunker
from transformers import AutoTokenizer
from llama_index.core.schema import BaseNode, Document, TextNode
from docling.document_converter import DocumentConverter # Corrected import for DocumentConverter

# Import custom file converter module
from file_converter import convert_file_for_upload, is_pdf_file, get_base_filename_no_ext # New import for local helpers

# Set up logging
logging.basicConfig(
    level=logging.DEBUG, # Set to DEBUG for more detailed logs during startup/troubleshooting
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Configuration settings
EMBED_MODEL_NAME = os.getenv("EMBED_MODEL_NAME", "BAAI/bge-en-icl") # Changed default
# HF_EMBEDDING_ENDPOINT_URL = os.getenv("HF_EMBEDDING_ENDPOINT_URL") # No longer needed

LLM_MODEL_NAME = os.getenv("LLM_MODEL_NAME", "mistralai/Mistral-Nemo-Instruct-2407") # Identifier for the model
# HF_LLM_ENDPOINT_URL = os.getenv("HF_LLM_ENDPOINT_URL") # No longer needed
NEBIUS_BASE_URL = "https://api.studio.nebius.com/v1/" # Fixed Nebius URL
NEBIUS_API_KEY = os.getenv("NEBIUS_API_KEY") # New environment variable for Nebius key

# HF_TOKEN = os.getenv("HF_TOKEN") # No longer needed

DOCUMENT_SET_DIR = os.getenv("DOCUMENT_SET_DIR", "./document_sets")
FRONTEND_URL = os.getenv("FRONTEND_URL", "*")
BACKEND_URL = os.getenv("REACT_APP_BACKEND_URL") or os.getenv("BACKEND_URL", "http://localhost:8000")
DOCLING_CHUNK_SIZE = int(os.getenv("DOCLING_CHUNK_SIZE", 1024))
DOCLING_CHUNK_OVERLAP = int(os.getenv("DOCLING_CHUNK_OVERLAP", 200))
RAG_NUM_CANDIDATES = int(os.getenv("RAG_NUM_CANDIDATES", 30))

# --- NOTE: Model initialization is MOVED to the lifespan function below ---
# Settings.llm and Settings.embed_model will be set during app startup.

# Print environment variables for debugging startup (useful on Render)
logger.info("--- Configuration ---")
logger.info(f"FRONTEND_URL: {FRONTEND_URL}")
logger.info(f"BACKEND_URL: {BACKEND_URL}")
logger.info(f"DOCUMENT_SET_DIR: {DOCUMENT_SET_DIR}")
logger.info(f"Embedding Model Name: {EMBED_MODEL_NAME}")
# logger.info(f"Embedding API URL: {HF_EMBEDDING_ENDPOINT_URL}") # Removed
logger.info(f"Embedding API Base URL: {NEBIUS_BASE_URL}") # Added
logger.info(f"LLM Model Name: {LLM_MODEL_NAME}")
# logger.info(f"LLM API URL: {HF_LLM_ENDPOINT_URL}") # Removed
logger.info(f"LLM API Base URL: {NEBIUS_BASE_URL}")
# logger.info(f"HF Token Provided: {'Yes' if HF_TOKEN else 'No'}") # Removed
logger.info(f"Nebius API Key Provided: {'Yes' if NEBIUS_API_KEY else 'No'}")
logger.info(f"Docling Chunk Size: {DOCLING_CHUNK_SIZE}")
logger.info(f"Docling Chunk Overlap: {DOCLING_CHUNK_OVERLAP}")
logger.info(f"RAG Candidates: {RAG_NUM_CANDIDATES}")
logger.info("--- End Configuration ---")

# --- Custom Nebius Embedding Class ---
class NebiusEmbedding(BaseEmbedding):
    """Custom LlamaIndex Embedding class using the standard OpenAI client for Nebius."""
    _client: StandardOpenAI = None
    _model_name: str = None

    def __init__(
        self,
        model_name: str,
        api_key: str,
        api_base: str,
        timeout: float = 120.0, # Default timeout
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs) # Pass model_name implicitly via kwargs?
        self._model_name = model_name
        try:
            self._client = StandardOpenAI(
                api_key=api_key,
                base_url=api_base,
                timeout=timeout,
            )
            logger.info(f"Custom NebiusEmbedding initialized for model '{model_name}' at '{api_base}'")
        except Exception as e:
            logger.error(f"Failed to initialize OpenAI client within NebiusEmbedding: {e}", exc_info=True)
            raise e

    def _get_text_embedding(self, text: str) -> Embedding:
        """Get text embedding for a single text entry."""
        if not self._client:
            raise ValueError("OpenAI client not initialized.")

        try:
            response = self._client.embeddings.create(input=[text], model=self._model_name)
            # Assuming response structure matches OpenAI standard:
            # response.data[0].embedding
            if response.data and len(response.data) > 0 and response.data[0].embedding:
                return response.data[0].embedding
            else:
                logger.error(f"Received unexpected embedding response format from Nebius: {response}")
                raise ValueError("Invalid embedding response received")
        except Exception as e:
            logger.error(f"Error getting embedding from Nebius for text '{text[:50]}...': {e}", exc_info=True)
            raise e # Re-raise the exception

    def _get_text_embedding_batch(
        self,
        texts: List[str],
        show_progress: bool = False,
    ) -> List[Embedding]:
        """Get text embeddings for a batch of texts."""
        # Note: Nebius API docs don't explicitly confirm batching in embeddings.create,
        # but the OpenAI spec supports it. We assume Nebius follows this.
        # If batching fails, we might need to implement simple iteration here.
        if not self._client:
            raise ValueError("OpenAI client not initialized.")

        try:
            response = self._client.embeddings.create(input=texts, model=self._model_name)
            # Assuming response structure matches OpenAI standard:
            # [item.embedding for item in response.data]
            if response.data and len(response.data) == len(texts):
                return [item.embedding for item in response.data if item.embedding]
            else:
                logger.error(f"Received unexpected batch embedding response format from Nebius: {response}")
                raise ValueError("Invalid batch embedding response received or length mismatch")
        except Exception as e:
            logger.error(f"Error getting batch embeddings from Nebius: {e}", exc_info=True)
            raise e # Re-raise the exception

    # --- Implement required query embedding methods --- 
    def _get_query_embedding(self, query: str) -> Embedding:
        """Get query embedding."""
        # For Nebius (and most embedding models), query embedding is the same as text embedding
        logger.debug(f"Getting query embedding for query: '{query[:50]}...'" )
        return self._get_text_embedding(query)
    
    # --- Implement required async methods (can fallback to sync if needed) --- 
    async def _aget_text_embedding(self, text: str) -> Embedding:
        """Asynchronously get text embedding. Falls back to sync for now."""
        # TODO: Implement true async using an async OpenAI client if performance requires it
        logger.debug("Using synchronous fallback for _aget_text_embedding")
        return self._get_text_embedding(text)

    async def _aget_text_embedding_batch(self, texts: List[str]) -> List[Embedding]:
        """Asynchronously get text embeddings for a batch. Falls back to sync for now."""
        # TODO: Implement true async using an async OpenAI client if performance requires it
        logger.debug("Using synchronous fallback for _aget_text_embedding_batch")
        return self._get_text_embedding_batch(texts)

    async def _aget_query_embedding(self, query: str) -> Embedding:
        """Asynchronously get query embedding. Falls back to sync for now."""
        # TODO: Implement true async using an async OpenAI client if performance requires it
        logger.debug("Using synchronous fallback for _aget_query_embedding")
        return self._get_query_embedding(query) # Calls the sync query method

# --- Pydantic Models ---
class RagQueryRequest(BaseModel):
    set_name: str
    question: str

class ChunkPosition(BaseModel):
    pageNumber: int
    bbox: Optional[Dict[str, float]] = None
    coord_origin: Optional[str] = "BOTTOMLEFT"

class ChunkResponse(BaseModel):
    id: str
    text: str
    pdfUrl: str
    position: ChunkPosition
    pageNumber: int # Keep for compatibility if needed
    metadata: dict = {}

class RagAnswerResponse(BaseModel):
    answer: str
    chunks: List[ChunkResponse]

class DocumentSetResponse(BaseModel):
    set_name: str
    message: str

class DocumentSetListResponse(BaseModel):
    set_names: List[str]

# --- Helper Functions ---

def sanitize_name(name: str) -> str:
    """Sanitize a string to be safe for directory/collection names."""
    name = re.sub(r"[^a-zA-Z0-9_\-\.]", "_", name)
    name = re.sub(r"_+", "_", name)
    name = name.strip("_.")
    if not name:
        name = f"set_{uuid.uuid4().hex[:8]}"
    return name.lower()

def get_set_base_path(sanitized_set_name: str) -> str:
    """Get the base path for a document set."""
    return os.path.join(DOCUMENT_SET_DIR, sanitized_set_name)

def get_set_docs_path(sanitized_set_name: str) -> str:
    """Get the path to the documents directory for a set."""
    return os.path.join(get_set_base_path(sanitized_set_name), "documents")

def get_set_chroma_path(sanitized_set_name: str) -> str:
    """Get the path for ChromaDB persistence for a set."""
    return os.path.join(get_set_base_path(sanitized_set_name), "chroma_db")

def get_chroma_storage_context(sanitized_set_name: str) -> StorageContext:
    """
    Initializes ChromaDB client and returns a StorageContext for indexing a set.
    Ensures the chroma path is clean before initialization.
    """
    chroma_path = get_set_chroma_path(sanitized_set_name)
    logger.info(f"Configuring ChromaDB storage context for set '{sanitized_set_name}' at path: {chroma_path}")

    # --- Robust cleanup ---
    if os.path.exists(chroma_path):
        logger.warning(f"Existing ChromaDB path found at '{chroma_path}'. Attempting removal.")
        try:
            shutil.rmtree(chroma_path)
            # time.sleep(0.1) # Optional pause
            if os.path.exists(chroma_path):
                # This shouldn't happen if rmtree succeeded without error
                logger.error(f"ChromaDB path '{chroma_path}' still exists after rmtree attempt without error. Filesystem issue? Check permissions or locks.")
                raise HTTPException(status_code=500, detail=f"Failed to properly clean existing storage path: {chroma_path}")
            else:
                logger.info(f"Successfully removed existing ChromaDB path: {chroma_path}")
        except PermissionError as pe:
            logger.error(f"PermissionError removing existing ChromaDB directory '{chroma_path}': {pe}", exc_info=True)
            raise HTTPException(status_code=500, detail=f"Permission denied cleaning storage path: {pe}")
        except Exception as e_rm: # Catch broader exceptions during cleanup
            logger.error(f"Unexpected error removing existing ChromaDB directory '{chroma_path}': {e_rm}", exc_info=True)
            raise HTTPException(status_code=500, detail=f"Failed to clean up existing storage path: {e_rm}")

    # --- Proceed with creating the directory and initializing ---
    try:
        logger.debug(f"Attempting os.makedirs for '{chroma_path}'")
        os.makedirs(chroma_path, exist_ok=True)
        logger.info(f"Ensured ChromaDB directory exists: {chroma_path}")
        # Verify directory creation and permissions
        if not os.path.isdir(chroma_path):
             raise OSError(f"Failed to create directory or path is not a directory: {chroma_path}")
        # Add a basic write check
        try:
            test_file_path = os.path.join(chroma_path, ".write_test")
            with open(test_file_path, "w") as f_test:
                f_test.write("test")
            os.remove(test_file_path)
            logger.debug(f"Write test successful in {chroma_path}")
        except Exception as write_err:
            logger.error(f"Write test failed in {chroma_path}. Check permissions. Error: {write_err}", exc_info=True)
            raise HTTPException(status_code=500, detail=f"Directory write permission issue in {chroma_path}")

        collection_name = f"set_{sanitized_set_name}"
        if len(collection_name) < 3: collection_name = f"{collection_name}_coll"
        if len(collection_name) > 63: collection_name = collection_name[:63]
        if not re.match(r"^[a-zA-Z0-9][a-zA-Z0-9._-]{1,61}[a-zA-Z0-9]$", collection_name):
             collection_name = re.sub(r'[^a-zA-Z0-9._-]', '_', collection_name)
             collection_name = re.sub(r'^[a-zA-Z0-9]+', '', collection_name)
             collection_name = re.sub(r'[^a-zA-Z0-9]+$', '', collection_name)
             if len(collection_name) < 3: collection_name = f"col_{collection_name}"[:63]
             if len(collection_name) > 63: collection_name = collection_name[:63]
             logger.warning(f"Sanitized collection name further to: '{collection_name}'")

        # Initialize the client - this is where the original error occurred
        logger.debug(f"Attempting chromadb.PersistentClient(path='{chroma_path}')")
        # Add extra logging just before the critical call
        logger.info(">>> Initializing chromadb.PersistentClient...")
        db = chromadb.PersistentClient(path=chroma_path)
        logger.info("<<< chromadb.PersistentClient initialized.")
        logger.debug(f"PersistentClient initialized for path: {chroma_path}")

        logger.debug(f"Getting or creating collection: '{collection_name}'")
        chroma_collection = db.get_or_create_collection(collection_name)
        logger.debug(f"Collection '{collection_name}' obtained.")

        vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
        # We still might need a docstore for metadata or relationships beyond Chroma's capabilities
        docstore = SimpleDocumentStore()

        storage_context = StorageContext.from_defaults(
            vector_store=vector_store,
            docstore=docstore,
            # No index_store needed here for basic Chroma setup
        )
        logger.info(f"ChromaDB Storage Context configured successfully for set '{sanitized_set_name}'.")
        return storage_context

    except HTTPException: # Re-raise explicit HTTP exceptions
        raise
    except Exception as e:
        # Update the error logging here as well
        logger.error(f"Failed to configure storage context for set '{sanitized_set_name}' after directory setup: {e}", exc_info=True)
        if "Could not connect to tenant default_tenant" in str(e) or "no such table: tenants" in str(e):
             logger.error("PersistentClient init failed. This is unexpected after cleaning/testing the directory. "
                          f"Check filesystem/permissions for '{chroma_path}'. Consider checking ChromaDB/sqlite3 compatibility or reporting a ChromaDB bug.")
        raise HTTPException(status_code=500, detail=f"Storage context configuration failed: {e}")
    
def load_set_index(sanitized_set_name: str) -> Tuple[VectorStoreIndex, ChromaVectorStore]:
    """Load the index and the vector store instance directly from ChromaDB."""
    chroma_path = get_set_chroma_path(sanitized_set_name)
    logger.info(f"Attempting to load index for set '{sanitized_set_name}' directly from ChromaDB at: {chroma_path}")

    # <<< Check Models Initialized (Directly within Load) >>>
    # This check is crucial because load_set_index might be called before lifespan finishes
    # or if lifespan failed.
    if not Settings.embed_model or not Settings.llm:
        logger.error(f"Attempted to load index for '{sanitized_set_name}' but backend models are not available.")
        raise HTTPException(status_code=503, detail="Backend models not available. Check configuration and server logs.")
    # <<< End Check Models Initialized >>>

    try:
        vector_store = None
        collection_name = f"set_{sanitized_set_name}"
        if len(collection_name) < 3: collection_name = f"{collection_name}_coll"
        if len(collection_name) > 63: collection_name = collection_name[:63]

        if not os.path.exists(chroma_path):
             logger.error(f"ChromaDB path not found for set '{sanitized_set_name}': {chroma_path}")
             raise HTTPException(status_code=404, detail=f"Vector store not found for set '{sanitized_set_name}'. It might not be created.")

        try:
             db = chromadb.PersistentClient(path=chroma_path)
             chroma_collection = db.get_collection(collection_name) # Raises if not found
             if chroma_collection.count() == 0:
                 logger.error(f"ChromaDB collection '{collection_name}' exists but is empty for set '{sanitized_set_name}'.")
                 raise HTTPException(status_code=404, detail=f"Vector store for set '{sanitized_set_name}' is empty.")

             vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
             logger.info(f"Successfully connected to existing ChromaDB collection '{collection_name}' for set '{sanitized_set_name}'.")
        except Exception as chroma_err:
             logger.error(f"Failed to connect to or validate ChromaDB collection for set '{sanitized_set_name}': {chroma_err}", exc_info=True)
             raise HTTPException(status_code=404, detail=f"Vector store not found or invalid for set '{sanitized_set_name}': {chroma_err}")

        logger.info(f"Attempting VectorStoreIndex.from_vector_store for set {sanitized_set_name}")
        # LlamaIndex uses the globally configured Settings.embed_model here automatically
        index = VectorStoreIndex.from_vector_store(vector_store=vector_store)
        logger.info(f"Successfully loaded index from ChromaDB vector store for set {sanitized_set_name}")
        return index, vector_store

    except HTTPException:
         raise # Re-raise HTTP exceptions
    except Exception as e:
        logger.error(f"Unexpected error loading index from vector store for set '{sanitized_set_name}': {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to load index for set '{sanitized_set_name}': {e}")

def store_uploaded_files(sanitized_set_name: str, files: List[UploadFile]) -> List[str]:
    """
    Stores uploaded files in the set's documents directory.
    Converts non-PDF files to PDF format before saving the final version.
    Ensures final filenames do not have the 'temp_' prefix.
    """
    docs_path = get_set_docs_path(sanitized_set_name)
    os.makedirs(docs_path, exist_ok=True)
    final_file_paths = [] # Changed variable name for clarity
    conversion_stats = {"converted": 0, "already_pdf": 0, "failed": 0}

    for file in files:
        original_filename = file.filename
        if not original_filename:
            logger.warning(f"Skipping file with no filename.")
            continue

        temp_file_path = os.path.join(docs_path, f"temp_{original_filename}")
        converted_pdf_temp_path: Optional[str] = None # To store path from converter

        try:
            # 1. Save original to temp location
            with open(temp_file_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
            if not os.path.exists(temp_file_path) or os.path.getsize(temp_file_path) == 0:
                logger.error(f"Failed to save file to temp location or file is empty: {temp_file_path}")
                conversion_stats["failed"] += 1
                continue

            # 2. Check ORIGINAL filename extension
            is_original_pdf = original_filename.lower().endswith('.pdf')

            if is_original_pdf:
                # Already PDF: Define final path and move temp file to it
                final_filename = original_filename
                final_file_path = os.path.join(docs_path, final_filename)
                logger.info(f"Original file '{original_filename}' is PDF. Moving temp file to final path.")
                shutil.move(temp_file_path, final_file_path)
                final_file_paths.append(final_file_path)
                conversion_stats["already_pdf"] += 1
                temp_file_path = None # Mark temp file as handled

            else:
                # Not PDF: Attempt conversion
                logger.info(f"Original file '{original_filename}' is not PDF. Attempting conversion.")
                # Pass the temp file path to the converter
                success, converted_pdf_temp_path, _ = convert_file_for_upload(temp_file_path)

                if success and converted_pdf_temp_path:
                    # Conversion successful: Determine final filename and move converted temp file
                    original_base_no_ext = get_base_filename_no_ext(original_filename)
                    final_filename = f"{original_base_no_ext}.pdf" # Final name based on original
                    final_file_path = os.path.join(docs_path, final_filename)
                    logger.info(f"Conversion successful. Moving converted temp file '{converted_pdf_temp_path}' to final path '{final_file_path}'.")
                    shutil.move(converted_pdf_temp_path, final_file_path)
                    final_file_paths.append(final_file_path)
                    conversion_stats["converted"] += 1
                    # The temp dir containing converted_pdf_temp_path should be cleaned up by converter or here
                    # Let's add explicit cleanup of the *directory* where the converted temp PDF was stored
                    temp_conv_dir = os.path.dirname(converted_pdf_temp_path)
                    if temp_conv_dir and os.path.exists(temp_conv_dir) and tempfile.gettempdir() in temp_conv_dir:
                         logger.debug(f"Cleaning up conversion temp directory: {temp_conv_dir}")
                         shutil.rmtree(temp_conv_dir, ignore_errors=True)

                else:
                    # Conversion failed
                    logger.error(f"Failed to convert non-PDF file: {original_filename}")
                    conversion_stats["failed"] += 1
                    # Temp original file will be cleaned up in finally block

        except Exception as e:
            logger.error(f"Error processing file {original_filename}: {e}", exc_info=True)
            conversion_stats["failed"] += 1
        finally:
            # 3. Cleanup: Remove original temp file if it still exists
            if temp_file_path and os.path.exists(temp_file_path):
                logger.debug(f"Cleaning up original temp file: {temp_file_path}")
                os.remove(temp_file_path)
            # Cleanup potential lingering conversion temp dir if an exception occurred after conversion but before move
            if converted_pdf_temp_path:
                temp_conv_dir = os.path.dirname(converted_pdf_temp_path)
                if temp_conv_dir and os.path.exists(temp_conv_dir) and tempfile.gettempdir() in temp_conv_dir:
                    logger.debug(f"Ensuring cleanup of conversion temp directory after potential error: {temp_conv_dir}")
                    shutil.rmtree(temp_conv_dir, ignore_errors=True)


    logger.info(f"File processing statistics for set '{sanitized_set_name}': "
                f"{conversion_stats['converted']} files converted to PDF, "
                f"{conversion_stats['already_pdf']} files already in PDF format, "
                f"{conversion_stats['failed']} files failed processing.")

    return final_file_paths # Return list of final paths

def process_and_index_files(sanitized_set_name: str, file_paths: List[str]) -> bool:
    """Processes files using Docling Converter and HybridChunker, creates nodes manually, and indexes them."""
    # <<< Check Models Directly >>>
    if not Settings.embed_model or not Settings.llm:
        logger.error("Models not initialized via Settings. Cannot process and index files.")
        # Don't raise HTTP Exception here, just return False as it runs in background
        return False
    # <<< End Check Models Directly >>>

    if not file_paths:
        logger.error(f"No file paths provided for indexing set {sanitized_set_name}.")
        return False

    try:
        tokenizer = None
        # --- Use the tokenizer associated with the embedding model ---
        tokenizer_name = EMBED_MODEL_NAME # Use the same name as the embedding model
        logger.info(f"Attempting to initialize tokenizer '{tokenizer_name}' for HybridChunker.")
        try:
            # Attempt to download and load the specified tokenizer
            tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
            logger.info(f"Successfully initialized tokenizer '{tokenizer_name}'.")
        except Exception as e_tok:
             # Log error if specific tokenizer fails, fallback is removed for now
             logger.error(f"Failed to initialize specified tokenizer '{tokenizer_name}': {e_tok}. HybridChunker might use its default or fail if tokenizer is strictly required.", exc_info=True)
             tokenizer = None # Let HybridChunker handle potential missing tokenizer based on its implementation

        # Initialize HybridChunker - pass the loaded tokenizer if successful
        if tokenizer:
            hybrid_chunker = HybridChunker(tokenizer=tokenizer, max_tokens=DOCLING_CHUNK_SIZE, merge_peers=True)
            logger.info(f"Initialized HybridChunker with tokenizer '{tokenizer_name}' and max_tokens={DOCLING_CHUNK_SIZE}")
        else:
            # If tokenizer failed to load, initialize HybridChunker without one.
            # It might have internal defaults or require a tokenizer. Check HybridChunker docs if issues arise.
            hybrid_chunker = HybridChunker(max_tokens=DOCLING_CHUNK_SIZE, merge_peers=True)
            logger.warning(f"Initialized HybridChunker without a specific tokenizer (using defaults) due to previous error. max_tokens={DOCLING_CHUNK_SIZE}")


        storage_context = get_chroma_storage_context(sanitized_set_name)
        all_processed_nodes: List[BaseNode] = []
        processed_file_count = 0
        failed_file_count = 0
        converter = DocumentConverter()

        for file_path in file_paths:
            if not os.path.exists(file_path):
                logger.warning(f"File not found, skipping: {file_path}")
                failed_file_count += 1
                continue
            try:
                logger.info(f"Converting document: {file_path}...")
                conversion_result = converter.convert(source=file_path)
                if not conversion_result or not conversion_result.document:
                     logger.error(f"Docling conversion failed or returned no document for: {file_path}")
                     failed_file_count += 1
                     continue
                docling_doc_obj = conversion_result.document
                doc_node_count = 0
                logger.info(f"Chunking structured document from {file_path} with HybridChunker...")
                raw_chunks = hybrid_chunker.chunk(dl_doc=docling_doc_obj)

                for i, chunk_data in enumerate(raw_chunks):
                    try:
                        chunk_text = getattr(chunk_data, 'text', str(chunk_data))
                        chunk_id_base = f"chunk_{i}"
                        metadata = {"filename": os.path.basename(file_path)}
                        page_no = 1
                        coord_origin = "BOTTOMLEFT"
                        bbox_json_str = "{}"
                        try:
                            meta = getattr(chunk_data, 'meta', None)
                            doc_items = getattr(meta, 'doc_items', []) if meta else []
                            if doc_items:
                                first_doc_item = doc_items[0]
                                prov_list = getattr(first_doc_item, 'prov', [])
                                if prov_list:
                                    first_prov = prov_list[0]
                                    page_no = getattr(first_prov, 'page_no', 1)
                                    coord_origin_enum = getattr(first_prov, 'coord_origin', None)
                                    coord_origin = str(coord_origin_enum.value) if coord_origin_enum and hasattr(coord_origin_enum, 'value') else "BOTTOMLEFT"
                                    bbox_obj = getattr(first_prov, 'bbox', None)
                                    if bbox_obj:
                                        bbox_dict = {}
                                        for attr in ['l', 't', 'r', 'b', 'x0', 'y0', 'x1', 'y1', 'top', 'left', 'bottom', 'right', 'width', 'height']:
                                            if hasattr(bbox_obj, attr):
                                                val = getattr(bbox_obj, attr)
                                                try: bbox_dict[str(attr)] = float(val)
                                                except (ValueError, TypeError): bbox_dict[str(attr)] = str(val)
                                        if bbox_dict:
                                            try: bbox_json_str = json.dumps(bbox_dict)
                                            except Exception as json_err: logger.warning(f"Error serializing bbox dict for node {chunk_id_base}: {json_err}")
                        except Exception as prov_err:
                            logger.warning(f"Error extracting provenance for chunk {i} in {file_path}: {prov_err}", exc_info=True)
                        metadata["page_no"] = page_no
                        metadata["coord_origin"] = coord_origin
                        metadata["bbox_json"] = bbox_json_str
                        node = TextNode(
                            text=chunk_text,
                            id_=f"{sanitized_set_name}_{os.path.basename(file_path)}_{chunk_id_base}",
                            metadata=metadata,
                        )
                        all_processed_nodes.append(node)
                        doc_node_count += 1
                    except Exception as node_creation_err:
                        logger.error(f"Error creating TextNode for chunk {i} in {file_path}: {node_creation_err}", exc_info=True)

                if doc_node_count > 0:
                     processed_file_count += 1
                     logger.info(f"Processed {doc_node_count} nodes via HybridChunker from {os.path.basename(file_path)}")
                else:
                    logger.warning(f"HybridChunker returned no processable chunks for document: {file_path}")
                    failed_file_count += 1
            except Exception as process_err:
                 logger.error(f"Failed to process file {file_path}: {process_err}", exc_info=True)
                 failed_file_count += 1

        if not all_processed_nodes:
            logger.error(f"Failed to generate any processable nodes for set {sanitized_set_name}. No index created.")
            return False

        logger.info(f"Creating/updating index from {len(all_processed_nodes)} total processed nodes for set {sanitized_set_name}... (Data goes to ChromaDB)")
        # Uses Settings.embed_model implicitly
        index = VectorStoreIndex(nodes=all_processed_nodes, storage_context=storage_context)
        logger.info(f"Successfully indexed nodes into ChromaDB for set '{sanitized_set_name}' with nodes from {processed_file_count} files.")
        if failed_file_count > 0:
             logger.warning(f"{failed_file_count} files failed to process for set '{sanitized_set_name}'.")
        return True

    except Exception as e:
        logger.error(f"Error during indexing process for set {sanitized_set_name}: {e}", exc_info=True)
        return False

def cosine_similarity(v1, v2):
    """Compute cosine similarity between two vectors."""
    if not isinstance(v1, (list, np.ndarray)) or not isinstance(v2, (list, np.ndarray)) or len(v1) == 0 or len(v2) == 0: return 0.0
    vec1 = np.array(v1).astype(float)
    vec2 = np.array(v2).astype(float)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    if norm1 == 0 or norm2 == 0: return 0.0
    similarity = np.dot(vec1, vec2) / (norm1 * norm2)
    return max(0.0, min(1.0, similarity)) # Clamp between 0 and 1

# --- FastAPI App and Lifespan ---

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Handles application startup and shutdown events.
    - Creates base directory.
    - Initializes LlamaIndex Settings (LLM and Embedding models).
    """
    logger.info("Lifespan: Startup sequence initiated.")

    # 1. Ensure base directory exists
    if not os.path.exists(DOCUMENT_SET_DIR):
        try:
            os.makedirs(DOCUMENT_SET_DIR)
            logger.info(f"Created base directory: {DOCUMENT_SET_DIR}")
        except Exception as e:
            logger.error(f"FATAL: Failed to create base directory {DOCUMENT_SET_DIR}: {e}", exc_info=True)
            raise RuntimeError(f"Failed to create required directory {DOCUMENT_SET_DIR}") from e

    # 2. Initialize models within lifespan to handle potential startup errors gracefully
    logger.info("Lifespan: Initializing external model clients...")
    # Ensure globals are accessible (though direct access is preferred if possible)
    global EMBED_MODEL_NAME, LLM_MODEL_NAME, NEBIUS_API_KEY, NEBIUS_BASE_URL

    try:
        # Validate required variables (check again inside lifespan for clarity)
        # if not HF_EMBEDDING_ENDPOINT_URL: raise ValueError("HF_EMBEDDING_ENDPOINT_URL environment variable not set.") # Removed
        # if not HF_LLM_ENDPOINT_URL: raise ValueError("HF_LLM_ENDPOINT_URL environment variable not set.") # Removed check
        # if not HF_TOKEN: raise ValueError("HF_TOKEN environment variable not set for Embedding model.") # Removed
        if not NEBIUS_API_KEY: raise ValueError("NEBIUS_API_KEY environment variable not set for LLM and Embedding models.")

        # Initialize Embedding Model via Custom Nebius Class
        logger.info(f"Initializing Embedding client ({EMBED_MODEL_NAME}) via Custom NebiusEmbedding Class: {NEBIUS_BASE_URL}")
        Settings.embed_model = NebiusEmbedding(
            model_name=EMBED_MODEL_NAME, # Pass the actual Nebius model name
            api_base=NEBIUS_BASE_URL,
            api_key=NEBIUS_API_KEY,
            timeout=120.0
        )
        # No need for a separate log here, constructor logs success/failure
        # logger.info("Embedding client initialized successfully.")

        # Initialize LLM via OpenAILike wrapper pointed at Nebius
        logger.info(f"Initializing LLM client ({LLM_MODEL_NAME}) via OpenAILike Wrapper API: {NEBIUS_BASE_URL}")
        Settings.llm = OpenAILike(
            model=LLM_MODEL_NAME,
            api_base=NEBIUS_BASE_URL,      # Use Nebius base URL
            api_key=NEBIUS_API_KEY,        # Use Nebius API Key
            is_chat_model=True,            # Explicitly state it's a chat model
            request_timeout=120.0          # Keep the timeout
            # additional_kwargs={"timeout": 120.0} # Timeout might be handled by request_timeout directly
        )
        logger.info("LLM client initialized successfully.")

        logger.info("Lifespan: All model clients initialized.")

    except Exception as e:
        logger.error(f"FATAL: Failed during model client setup in lifespan: {e}", exc_info=True)
        # Set models to None so health checks fail and endpoints raise 503
        Settings.embed_model = None
        Settings.llm = None
        logger.error("Models set to None due to initialization failure. Application will be unhealthy.")
        # Depending on policy, you might want to re-raise to stop the app:
        # raise RuntimeError("Model initialization failed during startup") from e

    # Yield control to the application running state
    logger.info("Lifespan: Startup complete. Application running.")
    yield
    # --- Cleanup phase (runs on shutdown) ---
    logger.info("Lifespan: Shutdown sequence initiated.")
    # Clear global settings
    Settings.llm = None
    Settings.embed_model = None
    logger.info("Lifespan: Model clients cleared.")
    logger.info("Lifespan: Shutdown sequence complete.")


# Create FastAPI app instance *AFTER* defining lifespan
app = FastAPI(lifespan=lifespan, title="Document Set Q&A API", description="API for managing and querying document sets.")

# Add CORS middleware
origins = [origin.strip() for origin in FRONTEND_URL.split(',')] if FRONTEND_URL != "*" else ["*"]
logger.info(f"Configuring CORS for origins: {origins}")
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["Content-Disposition", "Content-Type"],
)

# --- API Endpoints ---

@app.post("/document_sets", response_model=DocumentSetResponse, status_code=201)
async def create_document_set(name: str = Form(...), files: List[UploadFile] = File(...)):
    """
    Create a new document set by uploading files and processing them immediately.
    """
    # Use consistent variable name within this function scope
    sanitized_set_name = None # Initialize

    try:
        if not name: raise HTTPException(status_code=400, detail="Set name cannot be empty.")
        if not files: raise HTTPException(status_code=400, detail="At least one file must be uploaded.")

        # Assign to the consistent variable name
        sanitized_set_name = sanitize_name(name)
        # Pass the consistent variable name
        set_base_path = get_set_base_path(sanitized_set_name)
        chroma_path = get_set_chroma_path(sanitized_set_name)

        if os.path.exists(set_base_path) and os.path.exists(chroma_path) and os.listdir(chroma_path):
            logger.warning(f"Attempt to create existing and indexed set '{sanitized_set_name}'.")
            raise HTTPException(status_code=409, detail=f"Document set '{sanitized_set_name}' already exists and appears indexed.")
        elif os.path.exists(set_base_path):
             logger.warning(f"Directory for set '{sanitized_set_name}' exists but seems incomplete/unindexed. Proceeding to create/overwrite.")

        # Pass the consistent variable name
        saved_file_paths = store_uploaded_files(sanitized_set_name, files)
        if not saved_file_paths:
            raise HTTPException(status_code=400, detail="No valid files were uploaded or saved.")

        loop = asyncio.get_running_loop()
        try:
            logger.info(f"Starting background file processing and indexing for set '{sanitized_set_name}'...")
            # Pass the consistent variable name
            success = await loop.run_in_executor(
                None, process_and_index_files, sanitized_set_name, saved_file_paths
            )
            logger.info(f"Background processing/indexing finished for set '{sanitized_set_name}'. Success: {success}")
        except Exception as exec_err:
            logger.error(f"Error during background processing/indexing for set '{sanitized_set_name}': {exec_err}", exc_info=True)
            raise HTTPException(status_code=500, detail=f"Error during file processing: {exec_err}")

        if not success:
            raise HTTPException(status_code=500, detail=f"Failed to process and index files for set '{sanitized_set_name}'. Check server logs.")

        return DocumentSetResponse(
            set_name=sanitized_set_name, # Return the consistent variable name
            message=f"Document set '{sanitized_set_name}' created successfully with {len(saved_file_paths)} file(s)."
        )

    except HTTPException:
        raise

    except Exception as e:
        # --- Robust final exception handler (using consistent variable name) ---
        local_vars = locals()
        set_identifier = name if 'name' in local_vars and name else "unknown"
        # Access the potentially undefined variable safely using .get() and the consistent name
        log_identifier = local_vars.get('sanitized_set_name') if 'sanitized_set_name' in local_vars else set_identifier

        logger.error(f"Unexpected error creating document set '{log_identifier}': {e}", exc_info=True)

        # Construct detail message safely using consistent name
        detail_message = f"Internal server error creating set '{set_identifier}': {e}"
        current_sanitized_name = local_vars.get('sanitized_set_name') # Use consistent name
        if current_sanitized_name and current_sanitized_name != set_identifier:
            detail_message = f"Internal server error creating set '{set_identifier}' (sanitized as '{current_sanitized_name}'): {e}"

        raise HTTPException(status_code=500, detail=detail_message)
        # --- End robust handler ---

@app.get("/document_sets", response_model=DocumentSetListResponse)
async def list_document_sets():
    """List available document sets by directory name."""
    try:
        if not os.path.exists(DOCUMENT_SET_DIR):
            logger.warning(f"Document set directory not found: {DOCUMENT_SET_DIR}")
            return DocumentSetListResponse(set_names=[]) # Return empty list if base dir doesn't exist

        set_names = [
            d for d in os.listdir(DOCUMENT_SET_DIR)
            if os.path.isdir(os.path.join(DOCUMENT_SET_DIR, d))
        ]
        return DocumentSetListResponse(set_names=sorted(set_names))
    except Exception as e:
        logger.error(f"Error listing document sets: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Error listing document sets")

@app.get("/document_sets/{set_name}/documents/{filename}")
async def serve_set_pdf(set_name: str, filename: str):
    """Serve a PDF file from a specific document set."""
    sanitized_name = sanitize_name(set_name)
    try:
        decoded_filename = urllib.parse.unquote(filename)
        file_path = os.path.join(get_set_docs_path(sanitized_name), decoded_filename)
        logger.debug(f"Request to serve PDF: Set='{sanitized_name}', File='{decoded_filename}', Path='{file_path}'")

        if not os.path.exists(file_path) or not os.path.isfile(file_path):
            logger.error(f"PDF file not found or is not a file: {file_path}")
            raise HTTPException(status_code=404, detail="File not found in specified set")

        if os.path.getsize(file_path) == 0:
            logger.error(f"PDF file is empty: {file_path}")
            raise HTTPException(status_code=400, detail="File is empty")

        # Optional: Basic PDF header check (already done during upload, but safe to re-check)
        # with open(file_path, "rb") as f:
        #     if not f.read(5) == b"%PDF-":
        #         logger.error(f"File does not appear to be a PDF: {file_path}")
        #         raise HTTPException(status_code=400, detail="Invalid file format")

        return FileResponse(
            file_path,
            media_type="application/pdf",
            filename=decoded_filename, # Send original filename back
            headers={"Content-Disposition": f"inline; filename=\"{decoded_filename}\""}
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error serving PDF file '{filename}' from set '{sanitized_name}': {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Error serving PDF file")

@app.options("/document_sets/{set_name}/documents/{filename}")
async def options_set_pdf(set_name: str, filename: str):
    # Required for CORS preflight requests when frontend fetches PDFs
    return Response(status_code=204) # No Content response is typical for OPTIONS

@app.delete("/document_sets/{set_name}", status_code=200)
async def delete_document_set(set_name: str):
    """Delete an entire document set, including its documents and index data."""
    sanitized_name = sanitize_name(set_name)
    set_base_path = get_set_base_path(sanitized_name)
    logger.info(f"Attempting to delete document set: '{sanitized_name}' at path: {set_base_path}")

    if not os.path.exists(set_base_path) or not os.path.isdir(set_base_path):
        logger.error(f"Document set directory not found for deletion: {set_base_path}")
        raise HTTPException(status_code=404, detail=f"Document set '{sanitized_name}' not found.")

    try:
        shutil.rmtree(set_base_path)
        logger.info(f"Successfully deleted document set directory: {set_base_path}")
        return {"message": f"Document set '{sanitized_name}' deleted successfully."}
    except Exception as e:
        logger.error(f"Error deleting document set '{sanitized_name}': {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to delete document set: {e}")

@app.post("/rag", response_model=RagAnswerResponse)
async def rag(request: RagQueryRequest):
    """
    Perform a RAG query against a specific document set.
    Returns the answer and the source chunks used to generate it.
    """
    sanitized_name = sanitize_name(request.set_name)
    logger.info(f"RAG Query for set '{sanitized_name}': {request.question}")

    try:
        # Load the index for the specified set
        index, vector_store = load_set_index(sanitized_name)

        # --- Define the Prompt Template Here ---
        # This is where you customize the instructions given to the LLM.
        # Modify the text within the triple quotes below.
        # Use {context_str} where the retrieved document chunks should be inserted.
        # Use {query_str} where the user's question should be inserted.
        DEFAULT_TEXT_QA_PROMPT_TMPL = (
            "Context information is below.\n"
            "---------------------\n"
            "{context_str}\n"
            "---------------------\n"
            "Given the context information and not prior knowledge, "
            "answer the query concisely and factually. If the context does not contain the answer, state that you cannot answer based on the provided information.\n"
            "Always answer in the same language as the query.\n"
            "Query: {query_str}\n"
            "Answer: "
        )
        qa_template = PromptTemplate(DEFAULT_TEXT_QA_PROMPT_TMPL)
        # ------------------------------------

        # Create a query engine with the loaded index and custom prompt
        query_engine = index.as_query_engine(
            similarity_top_k=RAG_NUM_CANDIDATES,
            text_qa_template=qa_template, # Pass the custom template here
        )

        # Execute the query
        # --- Use await for async query ---
        response = await query_engine.aquery(request.question)
        # --- End change ---

        # Extract source nodes and their metadata
        source_nodes = getattr(response, 'source_nodes', [])

        # Prepare the response chunks
        chunks = []
        for node in source_nodes:
             try:
                 metadata = getattr(node, 'metadata', {})
                 node_text = getattr(node, 'text', '')
                 filename = metadata.get('filename', 'unknown.pdf')
                 page_no = metadata.get('page_no', 1)
                 coord_origin = metadata.get('coord_origin', 'BOTTOMLEFT')
                 bbox_dict = {}
                 bbox_json_str = metadata.get('bbox_json', '{}')
                 try:
                     if bbox_json_str and bbox_json_str != '{}':
                         bbox_dict = json.loads(bbox_json_str)
                 except json.JSONDecodeError:
                     logger.warning(f"Failed to parse bbox JSON: {bbox_json_str}")
                 safe_filename = urllib.parse.quote(filename)
                 base_url = BACKEND_URL.rstrip('/')
                 pdf_url = f"{base_url}/document_sets/{sanitized_name}/documents/{safe_filename}"
                 chunk = ChunkResponse(
                     id=getattr(node, 'id_', f"node_{len(chunks)}"),
                     text=node_text,
                     pdfUrl=pdf_url,
                     position=ChunkPosition(
                         pageNumber=page_no,
                         bbox=bbox_dict,
                         coord_origin=coord_origin
                     ),
                     pageNumber=page_no,
                     metadata=metadata
                 )
                 chunks.append(chunk)
             except Exception as chunk_err:
                 logger.error(f"Error processing source node: {chunk_err}", exc_info=True)

        return RagAnswerResponse(
             answer=str(response.response or "Could not generate an answer based on the provided documents."), # Get answer text from response object
             chunks=chunks
         )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error performing RAG query for set '{sanitized_name}': {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error performing RAG query: {e}")

@app.get("/health")
async def health_check():
    """
    Health check endpoint to verify the API is running and its components are healthy.
    Returns 200 OK if healthy, 503 Service Unavailable if unhealthy.
    """
    # Check if models are initialized
    llm_ok = Settings.llm is not None
    embed_ok = Settings.embed_model is not None
    
    # Check if document directory exists
    dir_ok = os.path.exists(DOCUMENT_SET_DIR) and os.path.isdir(DOCUMENT_SET_DIR)
    dir_error = None
    
    if not dir_ok:
        dir_error = f"Document directory not found or not a directory: {DOCUMENT_SET_DIR}"
        logger.warning(dir_error)
    
    # Compile status details
    status_details = {
        "llm_initialized": llm_ok,
        "embedding_model_initialized": embed_ok,
        "document_directory_ok": dir_ok,
    }
    if dir_error:
        status_details["directory_error"] = dir_error

    if llm_ok and embed_ok and dir_ok:
        logger.debug("Health check passed.")
        return {"status": "healthy", "details": status_details}
    else:
        logger.warning(f"Health check failed: {status_details}")
        # Return 503 Service Unavailable if any critical component is down
        return JSONResponse(
            status_code=503,
            content={"status": "unhealthy", "details": status_details}
        )

# --- NEW ENDPOINT ---
@app.get("/document_sets/{set_name}/files", response_model=List[str])
async def list_files_in_document_set(set_name: str):
    """List the final (PDF) files currently present in a document set's directory."""
    sanitized_set_name = sanitize_name(set_name)
    docs_path = get_set_docs_path(sanitized_set_name)

    if not os.path.isdir(docs_path):
        logger.error(f"Attempted to list files for non-existent set directory: {docs_path}")
        raise HTTPException(status_code=404, detail=f"Document set '{sanitized_set_name}' not found.")

    try:
        # List files, ensuring they are actually files (not directories)
        # We rely on the store_uploaded_files logic having produced the final .pdf names
        filenames = [
            f for f in os.listdir(docs_path)
            if os.path.isfile(os.path.join(docs_path, f)) and not f.startswith('temp_') # Exclude potential leftover temp files
        ]
        logger.info(f"Found {len(filenames)} file(s) in set '{sanitized_set_name}'.")
        return sorted(filenames) # Return sorted list
    except Exception as e:
        logger.error(f"Error listing files for set '{sanitized_set_name}': {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error listing files for set '{sanitized_set_name}'.")
# --- END NEW ENDPOINT ---

# --- Helper to Rebuild Index --- (NEW)
async def _rebuild_set_index(sanitized_set_name: str):
    """Helper function to rebuild the index for a set after changes."""
    logger.info(f"Starting index rebuild process for set '{sanitized_set_name}'...")
    docs_path = get_set_docs_path(sanitized_set_name)
    if not os.path.isdir(docs_path):
        logger.error(f"Cannot rebuild index: Documents directory not found for set '{sanitized_set_name}'.")
        return False # Indicate failure

    try:
        # Get all current PDF files in the directory
        current_files = [
            os.path.join(docs_path, f)
            for f in os.listdir(docs_path)
            if os.path.isfile(os.path.join(docs_path, f)) and f.lower().endswith('.pdf')
        ]

        if not current_files:
            logger.warning(f"No PDF files found in '{docs_path}' for set '{sanitized_set_name}'. Clearing index.")
            # If no files, ensure the Chroma DB is cleared/reset
            storage_context = get_chroma_storage_context(sanitized_set_name)
            # Potential issue: get_chroma_storage_context currently DELETES the dir.
            # We might need a gentler clear if the dir should persist empty.
            # For now, recreating it empty should be okay.
            logger.info(f"Index cleared/reset for empty set '{sanitized_set_name}'.")
            return True # Technically successful in representing the empty state

        logger.info(f"Found {len(current_files)} files to re-index for set '{sanitized_set_name}'.")
        # Use run_in_executor for the potentially long-running process_and_index_files
        # Ensure we run this synchronously within the helper context
        # loop = asyncio.get_running_loop()
        # success = await loop.run_in_executor(
        #     None, process_and_index_files, sanitized_set_name, current_files
        # )
        # Let's try running it directly if process_and_index_files is synchronous
        # If process_and_index_files becomes async itself, we can await it.
        # Assuming process_and_index_files is blocking/synchronous:
        success = process_and_index_files(sanitized_set_name, current_files)

        if success:
            logger.info(f"Successfully rebuilt index for set '{sanitized_set_name}'.")
            return True
        else:
            logger.error(f"Failed to rebuild index for set '{sanitized_set_name}'.")
            return False
    except Exception as e:
        logger.error(f"Error during index rebuild for set '{sanitized_set_name}': {e}", exc_info=True)
        return False
# --- End Helper ---

# --- DELETE Single File Endpoint --- (NEW)
@app.delete("/document_sets/{set_name}/documents/{filename}", status_code=202) # 202 Accepted as index rebuild runs in background
async def delete_file_from_document_set(set_name: str, filename: str):
    """Delete a specific file from a document set and trigger re-indexing."""
    sanitized_set_name = sanitize_name(set_name)
    try:
        # Decode filename just in case it's URL encoded (though usually path params aren't)
        decoded_filename = urllib.parse.unquote(filename)
    except Exception as decode_err:
        logger.error(f"Error decoding filename '{filename}': {decode_err}")
        raise HTTPException(status_code=400, detail=f"Invalid filename format: {filename}")

    docs_path = get_set_docs_path(sanitized_name)
    file_path = os.path.join(docs_path, decoded_filename)

    logger.info(f"Attempting to delete file: '{file_path}' from set '{sanitized_set_name}'")

    if not os.path.isfile(file_path):
        logger.error(f"File not found for deletion: {file_path}")
        # Return 404 even if the set exists but the file doesn't
        raise HTTPException(status_code=404, detail=f"File '{decoded_filename}' not found in set '{sanitized_set_name}'.")

    try:
        os.remove(file_path)
        logger.info(f"Successfully deleted file: {file_path}")
    except OSError as e:
        logger.error(f"Error deleting file '{file_path}': {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to delete file '{decoded_filename}': {e}")

    # Trigger background re-indexing AFTER successful deletion
    # Use asyncio.create_task to run the async helper in the background
    try:
        logger.info(f"Scheduling background index rebuild for set '{sanitized_set_name}' after deleting file '{decoded_filename}'...")
        asyncio.create_task(_rebuild_set_index(sanitized_set_name))
    except Exception as schedule_err:
        logger.error(f"Error scheduling background index rebuild for set '{sanitized_set_name}' after deletion: {schedule_err}", exc_info=True)
        # Return success for deletion, but warn about indexing
        return {"message": f"File '{decoded_filename}' deleted, but failed to schedule background re-indexing. Manual rebuild might be needed."}

    return {"message": f"File '{decoded_filename}' deleted successfully. Re-indexing started in the background."}
# --- END DELETE Single File Endpoint ---

# --- MODIFIED: Add Files Endpoint (Now uses _rebuild_set_index helper) ---
@app.post("/document_sets/{set_name}/add_files", status_code=202) # 202 Accepted for background task
async def add_files_to_document_set(set_name: str, files: List[UploadFile] = File(...)):
    # ...(existing code to check set exists, check files provided, call store_uploaded_files)... 
    sanitized_set_name = sanitize_name(set_name)
    set_base_path = get_set_base_path(sanitized_set_name)

    if not os.path.isdir(set_base_path):
        logger.error(f"Attempted to add files to non-existent set: {sanitized_set_name}")
        raise HTTPException(status_code=404, detail=f"Document set '{sanitized_set_name}' not found.")

    if not files:
        raise HTTPException(status_code=400, detail="At least one file must be provided.")

    try:
        logger.info(f"Adding {len(files)} file(s) to set '{sanitized_set_name}'...")
        newly_saved_paths = store_uploaded_files(sanitized_set_name, files)
        if not newly_saved_paths:
            raise HTTPException(status_code=400, detail="None of the provided files could be saved or processed.")
        logger.info(f"Successfully saved {len(newly_saved_paths)} new file(s) for set '{sanitized_set_name}'.")
    except Exception as store_err:
        logger.error(f"Error saving uploaded files for set '{sanitized_set_name}': {store_err}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error saving files: {store_err}")

    # Trigger background re-indexing USING THE HELPER
    # Use asyncio.create_task to run the async helper in the background
    try:
        logger.info(f"Scheduling background index rebuild for set '{sanitized_set_name}' after adding files...")
        asyncio.create_task(_rebuild_set_index(sanitized_set_name))
    except Exception as schedule_err:
        logger.error(f"Error scheduling background index rebuild for set '{sanitized_set_name}' after adding files: {schedule_err}", exc_info=True)
        # Return success for add, but warn about indexing
        return {"message": f"Files added to set '{sanitized_set_name}', but failed to schedule background re-indexing. Manual rebuild might be needed."}

    return {"message": f"Successfully added {len(newly_saved_paths)} file(s) to set '{sanitized_set_name}'. Re-indexing started in the background."}
# --- END MODIFIED Add Files ---

# --- Main execution block (for local development) ---
if __name__ == "__main__":
    import uvicorn
    # Use environment variables for host/port, defaulting for local dev
    host = os.getenv("HOST", "0.0.0.0") # Listen on all interfaces
    port = int(os.getenv("API_PORT", "8000")) # Use API_PORT, default 8000

    logger.info(f"--- Starting API Locally ({__name__}) ---")
    logger.info(f"Host: {host}")
    logger.info(f"Port: {port}")
    logger.info(f"Reload: True")
    logger.info("--- Access API at http://{host}:{port} ---")

    # Render's start command will use $PORT, this is just for local 'python api.py'
    uvicorn.run(
        "api:app", # Reference the FastAPI app instance within this file
        host=host,
        port=port,
        reload=True, # Enable auto-reload for development
        log_level="debug" # Set uvicorn log level for dev
        )
