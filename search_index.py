import logging
import torch
from langchain_huggingface import HuggingFaceEmbeddings # Changed import
from langchain_community.vectorstores import FAISS
from typing import List  # Import List from typing

# Set up logging with DEBUG level
logging.basicConfig(
    level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Configuration
INDEX_PATH = "data/index"  # Path to the saved FAISS index
EMBEDDING_MODEL = "all-MiniLM-L6-v2"  # Should match the model used for ingestion


def load_index(index_path: str, embedding_model_name: str) -> FAISS:
    """Loads the FAISS index from the specified path.

    Args:
        index_path: The path to the FAISS index.
        embedding_model_name:  The name of the Sentence Transformer model.

    Returns:
        The loaded FAISS index.  Returns None on error.
    """
    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.debug(f"Loading embeddings with model '{embedding_model_name}' on device '{device}'.")
        embeddings = HuggingFaceEmbeddings(  # Changed to use langchain_huggingface
            model_name=embedding_model_name,
            model_kwargs={"device": device},
        )
        logger.debug(f"Attempting to load FAISS index from: {index_path}")
        vector_store = FAISS.load_local(
            index_path, embeddings, allow_dangerous_deserialization=True
        )  # Added the allow_dangerous... argument
        logger.info(f"Loaded FAISS index from {index_path} with {vector_store.index.ntotal} vectors.")
        return vector_store
    except Exception as e:
        logger.error(f"Error loading FAISS index: {e}")
        return None


def search_index(
    vector_store: FAISS, query: str, search_kwargs: dict = {"k": 3}
) -> List[str]:
    """Searches the FAISS index for the most similar documents to the query.

    Args:
        vector_store: The FAISS index to search.
        query: The query string.
        search_kwargs:  Keyword arguments for the search.  "k" specifies
            the number of nearest neighbors to retrieve.

    Returns:
        A list of the retrieved documents (as strings).
        Returns an empty list on error or if the index is invalid.
    """
    if not vector_store:
        logger.error("Invalid FAISS index. Cannot perform search.")
        return []

    logger.debug(f"Searching index with query: '{query}' and kwargs: {search_kwargs}")
    try:
        results = vector_store.similarity_search(query, **search_kwargs)
        logger.debug(f"Found {len(results)} results.")
        # Extract the page_content.  similarity_search returns Document objects.
        return [result.page_content for result in results]
    except Exception as e:
        logger.error(f"Error searching index: {e}")
        return []


def main():
    """Main function to load the index and perform a search."""
    vector_store = load_index(INDEX_PATH, EMBEDDING_MODEL)
    if not vector_store:
        logger.error("Failed to load index. Exiting.")
        return

    query = "What is melting point?"  # Replace with your query
    logger.info(f"Performing search with query: '{query}'")
    results = search_index(vector_store, query)

    if results:
        logger.info(f"Search results for query: '{query}':")
        for i, result in enumerate(results):
            print(f"\nResult {i+1}:")
            print(result)
    else:
        logger.warning("No results found for the query.")


if __name__ == "__main__":
    main()