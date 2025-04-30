import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import os
import logging
from typing import List, Optional, Dict
from langchain_community.vectorstores import FAISS  # Import Langchain FAISS

class FaissIndexError(Exception):
    """Custom exception for FAISS index related errors."""
    pass

class ChunksFileError(Exception):
    """Custom exception for errors related to the chunks file."""
    pass

class TextbookAgent:
    def __init__(self, config: Optional[Dict] = None, faiss_index_path: str = "data/index", # Changed default path
                 chunks_file_path: str = "data/chunks.txt",
                 embedding_model_name: str = 'all-MiniLM-L6-v2',
                 embedding_model: Optional[SentenceTransformer] = None): # Accept pre-loaded embedding model
        """
        Initializes the TextbookAgent.

        Args:
            config (Optional[Dict]): Configuration dictionary. Defaults to None.
            faiss_index_path (str): Path to the FAISS index directory (for Langchain).
            chunks_file_path (str): Path to the text chunks file (will try to load from index if possible).
            embedding_model_name (str): Name of the SentenceTransformer model.
            embedding_model (Optional[SentenceTransformer]): Pre-loaded SentenceTransformer model.
        """
        self.config = config or {}  # Store the config
        self.index = None
        self.vectorstore = None  # To store the Langchain FAISS vectorstore
        if embedding_model is None:
            self.embedding_model = SentenceTransformer(embedding_model_name)
        else:
            self.embedding_model = embedding_model
        self.faiss_index_path = faiss_index_path
        self.chunks_file_path = chunks_file_path
        self.stored_chunks = [] # Initialize as empty, will try to load from index
        self._load_index()
        self.tokenizer = self.embedding_model.tokenizer  # Add the tokenizer
        self.dimension = self.embedding_model.get_sentence_embedding_dimension()  # Get dimension

    def _load_index(self):
        """Loads the Langchain FAISS index from the given path."""
        if not os.path.exists(self.faiss_index_path):
            logging.error(f"Langchain FAISS index directory not found at: {self.faiss_index_path}")
            raise FaissIndexError(f"Langchain FAISS index directory not found at: {self.faiss_index_path}. Ensure 'run_ingest.py' was executed.")

        try:
            logging.info(f"Loading existing Langchain FAISS index from: {self.faiss_index_path}")
            # Explicitly allow dangerous deserialization
            vectorstore = FAISS.load_local(self.faiss_index_path, self.embedding_model, allow_dangerous_deserialization=True)
            self.vectorstore = vectorstore
            self.index = vectorstore.index
            logging.info(f"Langchain FAISS index loaded from: {self.faiss_index_path}")
            # We can't directly know the number of chunks loaded here anymore
            # It will be determined during retrieval
            self.stored_chunks = None # Set to None, we'll retrieve chunks during search
        except Exception as e:
            logging.error(f"Error loading Langchain FAISS index: {e}")
            raise FaissIndexError(f"Error loading Langchain FAISS index: {e}") from e

        # Optionally, try to load chunks from the separate file as a fallback
        if not hasattr(self, 'vectorstore') and os.path.exists(self.chunks_file_path):
            try:
                with open(self.chunks_file_path, 'r', encoding='utf-8') as f:
                    self.stored_chunks = [line.strip() for line in f.readlines() if line.strip()]
                logging.info(f"Loaded {len(self.stored_chunks)} chunks from {self.chunks_file_path} (fallback).")
            except Exception as e:
                logging.warning(f"Error loading chunks from {self.chunks_file_path}: {e}")

    def get_relevant_chunks(self, query: str, top_k: int = 5) -> List[str]:
        """
        Retrieves the most relevant text chunks for a given query using FAISS.

        Args:
            query (str): The query string.
            top_k (int): The number of top-k relevant chunks to retrieve.

        Returns:
            List[str]: A list of the top-k relevant text chunks.
        """
        if not hasattr(self, 'vectorstore') or self.vectorstore is None:
            logging.warning("FAISS vectorstore not loaded. Returning empty list.")
            return []

        try:
            retriever = self.vectorstore.as_retriever(search_kwargs={'k': top_k})
            docs = retriever.get_relevant_documents(query)
            relevant_chunks = [doc.page_content for doc in docs]
            logging.info(f"Retrieved {len(relevant_chunks)} relevant chunks.")
            return relevant_chunks

        except Exception as e:
            logging.error(f"Error retrieving relevant chunks: {e}")
            return []

    def count_tokens(self, text: str) -> int:  # ADD THIS METHOD
        """Counts the number of tokens in the given text using the model's tokenizer."""
        return len(self.tokenizer.encode(text, add_special_tokens=False))