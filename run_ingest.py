import os
import logging
import torch
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from typing import List, Optional
from langchain.docstore.document import Document  # Import Document

# Set up logging with DEBUG level
logging.basicConfig(
    level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Configuration
DATA_PATH = "data/raw"  # Path to your raw PDF files
INDEX_PATH = "data/index"  # Path to save the FAISS index
EMBEDDING_MODEL = "all-MiniLM-L6-v2"  # Sentence Transformer model
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50


def load_documents(data_path: str) -> List[Document]:  # Return List[Document]
    """Loads PDF documents from the specified directory.

    Args:
        data_path: The path to the directory containing PDF files.

    Returns:
        A list of loaded documents. Returns an empty list on error.
    """
    documents = []
    if not os.path.exists(data_path):
        logger.error(f"Data path does not exist: {data_path}")
        return []

    for filename in os.listdir(data_path):
        if filename.endswith(".pdf"):
            filepath = os.path.join(data_path, filename)
            try:
                loader = PyPDFLoader(filepath)
                loaded_docs = loader.load()  # Load documents from the loader
                logger.debug(f"Loaded {len(loaded_docs)} documents from {filename}")
                if not loaded_docs:
                    logger.warning(f"No documents loaded from {filename}")
                documents.extend(loaded_docs)
                logger.info(f"Processed PDF: {filename}")
            except Exception as e:
                logger.error(f"Error loading {filename}: {e}")
                # Consider whether to continue processing other files
    if not documents:
        logger.warning("No PDF documents found in the specified directory.")
    return documents


def chunk_documents(
    documents: List[Document], chunk_size: int, chunk_overlap: int  # Expect List[Document]
) -> List[Document]:  # Return List[Document]
    """Splits the loaded documents into smaller, more manageable chunks.

    Args:
        documents: A list of documents to chunk.
        chunk_size: The maximum size of each chunk.
        chunk_overlap: The overlap between adjacent chunks.

    Returns:
        A list of text chunks (as Document objects). Returns an empty list on error.
    """
    if not documents:
        logger.warning("No documents to chunk.")
        return []

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
    try:
        chunks = text_splitter.split_documents(documents)
        logger.debug(f"Generated {len(chunks)} chunks from {len(documents)} documents.")
        if not chunks:
            logger.warning(
                "No chunks generated. Check document content and chunking parameters."
            )
            return []
        return chunks
    except Exception as e:
        logger.error(f"Error chunking documents: {e}")
        return []


def create_embeddings(
    embedding_model_name: str,
) -> Optional[HuggingFaceEmbeddings]:
    """Creates Hugging Face embeddings.

    Args:
        embedding_model_name: The name of the Sentence Transformer model.

    Returns:
        The HuggingFaceEmbeddings object. Returns None on error.
    """
    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Using PyTorch device: {device}")
        embeddings = HuggingFaceEmbeddings(
            model_name=embedding_model_name, model_kwargs={"device": device}
        )
        logger.debug(f"HuggingFaceEmbeddings created with model: {embedding_model_name} on device: {device}")
        return embeddings
    except Exception as e:
        logger.error(f"Error creating embeddings: {e}")
        return None


def create_index(
    chunks: List[Document], embeddings: HuggingFaceEmbeddings, index_path: str  # Expect List[Document]
) -> Optional[FAISS]:
    """Creates a FAISS index from the embeddings and saves it to disk.

    Args:
        chunks: The list of text chunks (as Document objects).
        embeddings: The HuggingFaceEmbeddings object.
        index_path: The path to save the FAISS index.

    Returns:
        The FAISS index. Returns None on error.
    """
    if not chunks:
        logger.warning("No chunks to index.")
        return None

    try:
        # Create the FAISS index from the documents and embeddings
        vector_store = FAISS.from_documents(chunks, embeddings)

        # Save the FAISS index locally
        vector_store.save_local(index_path)
        logger.debug(f"Created and saved FAISS index to {index_path} with {vector_store.index.ntotal} vectors.")
        return vector_store
    except Exception as e:
        logger.error(f"Error creating or saving FAISS index: {e}")
        return None


def main():
    """Main function to run the data ingestion process."""
    # Load documents
    documents = load_documents(DATA_PATH)
    if not documents:
        logger.error("No documents loaded. Ingestion process halted.")
        return

    # Chunk documents
    chunks = chunk_documents(documents, CHUNK_SIZE, CHUNK_OVERLAP)
    if not chunks:
        logger.error("Failed to chunk documents. Ingestion process halted.")
        return

    # Create embeddings
    embeddings = create_embeddings(EMBEDDING_MODEL)
    if not embeddings:
        logger.error("Failed to create embeddings. Ingestion process halted.")
        return

    # Create and save the FAISS index
    index = create_index(chunks, embeddings, INDEX_PATH)
    if not index:
        logger.error("Failed to create FAISS index. Ingestion process halted.")
        return

    logger.info("Data ingestion process completed successfully.")


if __name__ == "__main__":
    main()