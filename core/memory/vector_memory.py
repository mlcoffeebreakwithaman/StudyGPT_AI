import logging
import os
from typing import Optional, List

from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.docstore.document import Document

logger = logging.getLogger(__name__)

class VectorMemory:
    """
    Manages the vector store using Langchain FAISS.
    """
    def __init__(self, embedding_model_name: str = "all-MiniLM-L6-v2", index_path: str = "data/index"):
        self.embedding_model_name = embedding_model_name
        self.index_path = index_path
        self.embeddings = SentenceTransformerEmbeddings(model_name=self.embedding_model_name)
        self.vectorstore: Optional[FAISS] = self._load_index()

    def _load_index(self) -> Optional[FAISS]:
        """Loads the FAISS index from the specified path."""
        if os.path.exists(self.index_path):
            try:
                vectorstore = FAISS.load_local(
                    self.index_path, self.embeddings, allow_dangerous_deserialization=True
                )
                logger.info(f"Langchain FAISS index loaded from: {self.index_path}")
                return vectorstore
            except Exception as e:
                logger.error(f"Error loading FAISS index from {self.index_path}: {e}")
                return None
        else:
            logger.warning(f"FAISS index not found at: {self.index_path}. Ensure 'run_ingest.py' was executed.")
            return None

    def save_index(self, index_path: str = "data/index"):
        """Saves the current FAISS index to the specified path."""
        if self.vectorstore:
            try:
                self.vectorstore.save_local(index_path)
                logger.info(f"FAISS index saved to: {index_path}")
            except Exception as e:
                logger.error(f"Error saving FAISS index to {index_path}: {e}")
        else:
            logger.warning("No FAISS index to save.")

    def load_data(self, data_path: str = "data/raw"):
        """Loads data, chunks it, and creates the FAISS index."""
        from scripts.run_ingest import load_documents, chunk_documents, create_index

        documents = load_documents(data_path)
        if not documents:
            logger.warning("No documents loaded for indexing.")
            return

        chunks = chunk_documents(documents, chunk_size=500, chunk_overlap=50)
        if not chunks:
            logger.warning("No chunks generated for indexing.")
            return

        self.vectorstore = create_index(chunks, self.embeddings, self.index_path)
        if self.vectorstore:
            logger.info(f"FAISS index created with {self.vectorstore.index.ntotal} vectors.")

    async def retrieve(self, query: str, k: int = 3) -> List[Document]:
        """Retrieves the top k most relevant documents from the vector store."""
        if self.vectorstore:
            retriever = self.vectorstore.as_retriever(search_kwargs={'k': k})
            docs = await retriever.aget_relevant_documents(query)
            logger.info(f"VectorMemory: Retrieved {len(docs)} documents for query: '{query}'")
            return docs
        else:
            logger.warning("VectorMemory: Vector store not initialized. Cannot retrieve documents.")
            return []