import logging
from typing import Dict, Any, List

from langchain.embeddings import SentenceTransformerEmbeddings
from core.llm_wrapper import LLMWrapper
from core.memory.vector_memory import VectorMemory
from langchain.docstore.document import Document  # Import Document type
from core.agents.base_agent import BaseAgent
from core.agents.progress_agent import ProgressAgent

logger = logging.getLogger(__name__)

class TutorAgent(BaseAgent):
    """
    Agent responsible for answering questions based on the provided context.
    """
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.llm_wrapper: LLMWrapper = config.get("llm_wrapper")
        self.vector_memory: VectorMemory = config.get("vector_memory")
        self.max_new_tokens = 300
        self.top_k = 3
        self.embedding_model_name = "all-MiniLM-L6-v2" # Consider moving to config
        self.langchain_embeddings = SentenceTransformerEmbeddings(model_name=self.embedding_model_name)
        self._load_index()

    def _load_index(self):
        if self.vector_memory.vectorstore is not None:
            logger.info(f"TutorAgent: Langchain FAISS index loaded successfully from: {self.vector_memory.index_path}")
        else:
            logger.warning("TutorAgent: FAISS index not loaded.")

    async def get_relevant_chunks(self, query: str) -> str:
        """
        Retrieves the most relevant chunks from the vector store based on the query.

        Args:
            query (str): The query string.

        Returns:
            str: A string containing the combined relevant text chunks.
        """
        logger.info(f"TutorAgent: Getting relevant chunks for query: '{query}'")
        if self.vector_memory.vectorstore is None:
            logger.error("TutorAgent: FAISS vectorstore not loaded. Ensure data ingestion was successful.")
            return ""

        relevant_chunks = await self.vector_memory.retrieve(query, k=self.top_k)
        logger.info(f"TutorAgent: Retrieved relevant chunks: {relevant_chunks}")
        combined_chunks = "\n".join([doc.page_content for doc in relevant_chunks]) if relevant_chunks else ""
        logger.info(f"TutorAgent: Combined relevant chunks: '{combined_chunks}'")
        return combined_chunks

    def _create_prompt(self, question: str, context: str) -> str:
        return f"You are a helpful tutor. Answer the following question clearly and concisely, as if you are explaining it to a student. Use only the information provided in the following text. Do not add any information from outside the text.\n\nContext:\n{context}\n\nQuestion: {question}\n\nAnswer:"

    async def answer_question(self, question: str, context: str) -> str:
        logger.info(f"TutorAgent: Answering question: '{question}' with context: '{context}'")
        prompt = self._create_prompt(question, context)
        logger.info(f"TutorAgent: Generated prompt: '{prompt}'")
        try:
            response = self.llm_wrapper.generate_response(prompt, generation_kwargs={"max_new_tokens": self.max_new_tokens, "temperature": 0.7})
            logger.info(f"TutorAgent: Raw LLM response: {response}")
            answer = response[0] if response else "No answer found after generation."
            logger.info(f"TutorAgent: Generated answer: '{answer}'")
            return answer
        except Exception as e:
            logger.error(f"TutorAgent: Error during LLM generation: {e}")
            return "Sorry, I encountered an error while trying to generate an answer."

    async def handle_input(self, query: str, user_history: List[str] = None, progress: ProgressAgent = None) -> dict:
        logger.info(f"TutorAgent: Handling input: '{query}'")
        context = await self.get_relevant_chunks(query)
        if not context:
            return {"response": "I don't have relevant information to answer that."}
        answer = await self.answer_question(query, context)
        return {"response": answer}