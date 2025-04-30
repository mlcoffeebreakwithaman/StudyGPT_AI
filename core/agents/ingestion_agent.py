import pdfplumber
from sentence_transformers import SentenceTransformer
from nltk.tokenize import sent_tokenize
import numpy as np
import faiss
import os
import json
from typing import Optional, Dict
import logging
from core.exceptions import DataIngestionError # Import

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(name)s - %(message)s') # Modified format to include name
logger = logging.getLogger(__name__) # Get explicit logger instance

class IngestionAgent:
    def __init__(self, config: Optional[Dict] = None):
        """
        Initializes the IngestionAgent.
        """
        self.config = config or {}
        self.text_splitter = sent_tokenize
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.faiss_index = None
        self.chunks = []  # To store text chunks in order

    def load_document(self, file_path: str) -> str:
        """
        Loads the text content from a PDF document.

        Args:
            file_path (str): Path to the PDF file.

        Returns:
            str: The text content of the document.

        Raises:
            DataIngestionError: If the file is not a PDF or if an error occurs during reading.
        """
        if not file_path.lower().endswith('.pdf'):
            raise DataIngestionError(f"Unsupported file type: {file_path}. Only PDF files are supported.")
        try:
            with pdfplumber.open(file_path) as pdf:
                text = ""
                for page in pdf.pages:
                    text += page.extract_text() + "\n"  # Add newline between pages
                return text
        except Exception as e:
            raise DataIngestionError(f"Error reading PDF file: {file_path}. Details: {e}") from e

    def split_text(self, text: str) -> List[str]:
        """
        Splits the text into smaller chunks using sentence tokenization.

        Args:
            text (str): The text to split.

        Returns:
            List[str]: A list of text chunks.
        """
        if not text:
            raise DataIngestionError("Cannot split empty text.")
        try:
            chunks = self.text_splitter(text)
            return chunks
        except Exception as e:
            raise DataIngestionError(f"Error splitting text: {e}") from e

    def generate_embeddings(self, chunks: List[str]) -> np.ndarray:
        """
        Generates embeddings for the text chunks using the SentenceTransformer model.

        Args:
            chunks (List[str]): A list of text chunks.

        Returns:
            np.ndarray: A 2D numpy array containing the embeddings.
                        Each row represents the embedding for a chunk.

        Raises:
            DataIngestionError: If there are issues generating the embeddings.
        """
        if not chunks:
            raise DataIngestionError("Cannot generate embeddings for an empty list of chunks.")
        try:
            embeddings = self.embedding_model.encode(chunks)
            return embeddings
        except Exception as e:
            raise DataIngestionError(f"Error generating embeddings: {e}") from e

    def create_index(self, embeddings: np.ndarray, index_path: str) -> faiss.Index:
        """
        Creates a FAISS index from the embeddings and saves it to a file.

        Args:
            embeddings (np.ndarray): A 2D numpy array containing the embeddings.
            index_path (str): Path to save the FAISS index.

        Returns:
            faiss.Index: The created FAISS index.

        Raises:
            DataIngestionError: If there are issues creating or saving the index.
        """
        if not isinstance(embeddings, np.ndarray) or embeddings.ndim != 2:
            raise DataIngestionError(f"Invalid embeddings format. Expected a 2D numpy array, got {type(embeddings)} with shape {embeddings.shape if isinstance(embeddings, np.ndarray) else None}")

        if not embeddings.shape[0]:
            raise DataIngestionError("Cannot create index from empty embeddings.")

        dimension = embeddings.shape[1]
        try:
            index = faiss.IndexFlatIP(dimension)  # Use Inner Product for similarity
            index.add(embeddings)
            faiss.write_index(index, index_path)
            return index
        except Exception as e:
            raise DataIngestionError(f"Error creating or saving FAISS index: {e}") from e

    def process_document(self, file_path: str, index_path: str, chunks_file_path: str) -> None:
        """
        Processes the document, generating embeddings and creating a FAISS index.

        Args:
            file_path (str): Path to the document (PDF).
            index_path (str): Path to save the FAISS index.
            chunks_file_path (str): Path to save the text chunks.
        """
        try:
            text = self.load_document(file_path)
            chunks = self.split_text(text)
            embeddings = self.generate_embeddings(chunks)
            self.create_index(embeddings, index_path)
            self.chunks = chunks #store
            self._save_chunks(chunks, chunks_file_path)
            logger.info(f"Document processed. Index saved to {index_path}, chunks saved to {chunks_file_path}")

        except DataIngestionError as e:
            logger.error(f"Error processing document: {e}")
            raise  # Re-raise the DataIngestionError to be handled upstream

    def _save_chunks(self, chunks: List[str], chunks_file_path: str) -> None:
        """Saves the text chunks to a JSON file.

        Args:
            chunks (List[str]): The list of text chunks.
            chunks_file_path (str): The path to the file where chunks should be saved.
        Raises:
            DataIngestionError: If there is an error saving the chunks.
        """
        try:
            with open(chunks_file_path, 'w', encoding='utf-8') as f:
                json.dump(chunks, f, ensure_ascii=False, indent=4)
        except Exception as e:
            raise DataIngestionError(f"Error saving chunks to {chunks_file_path}: {e}") from e

    def load_chunks(self, chunks_file_path: str) -> List[str]:
        """Loads the text chunks from a JSON file.

        Args:
            chunks_file_path (str): The path to the file where chunks are saved.
        Returns:
            List[str]: The list of text chunks.
        Raises:
            DataIngestionError: If there is an error loading the chunks.
        """
        try:
            with open(chunks_file_path, 'r', encoding='utf-8') as f:
                chunks = json.load(f)
            return chunks
        except Exception as e:
            raise DataIngestionError(f"Error loading chunks from {chunks_file_path}: {e}") from e

if __name__ == '__main__':
    # Example usage
    ingestion_agent = IngestionAgent()
    pdf_file = "example.pdf"  # Replace with a real PDF file
    index_file = "data/index.faiss"
    chunks_file = "data/chunks.json" #added

    # Create a dummy PDF for testing if it doesn't exist
    if not os.path.exists(pdf_file):
        from reportlab.pdfgen import canvas
        c = canvas.Canvas(pdf_file)
        c.drawString(100, 750, "Hello, World!")
        c.save()
        logger.info(f"Created a dummy PDF file at {pdf_file}. Replace it with a real PDF for testing.")

    try:
        ingestion_agent.process_document(pdf_file, index_file, chunks_file)
        logger.info(f"Document processed. Index saved to {index_file}, chunks saved to {chunks_file}")
    except DataIngestionError as e:
        print(f"Error: {e}")

    # Load and print the chunks
    try:
        loaded_chunks = ingestion_agent.load_chunks(chunks_file)
        print(f"\nLoaded Chunks: {loaded_chunks}")
    except DataIngestionError as e:
        print(f"Error loading chunks: {e}")