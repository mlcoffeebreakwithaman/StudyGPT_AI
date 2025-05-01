# RecommendationAgent.py
import sys
import os
import yaml
from typing import List, Optional, Dict, Any
import logging
import random
import time

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(name)s - %(message)s')
logger = logging.getLogger(__name__)

# Add the project root directory to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, project_root)

from core.agents.base_agent import BaseAgent
from core.agents.textbook_agent import TextbookAgent, FaissIndexError, ChunksFileError
from core.llm_wrapper import LLMWrapper, LLMConnectionError, AgentConfigError
from core.exceptions import RecommendationAgentError  # Import


class RecommendationAgent(BaseAgent):
    def __init__(self, config: Optional[Dict[str, Any]] = None, **kwargs: Any):
        """
        Initializes the RecommendationAgent.

        Args:
            config (Optional[Dict]): Configuration dictionary containing llm_wrapper and vector_memory. Defaults to None.
            **kwargs (Any): Additional keyword arguments.
        """
        super().__init__(config, **kwargs)
        if config:
            self.llm_wrapper: LLMWrapper = config.get("llm_wrapper")
            self.vector_memory = config.get("vector_memory")
        else:
            raise ValueError("Configuration dictionary must be provided to RecommendationAgent.")

        self._initialize_internal_components(config)

        logger = logging.getLogger(__name__)
        logger.info("RecommendationAgent initialized.")

    def _initialize_internal_components(self, config: Optional[Dict[str, Any]] = None):
        """Initializes internal components like TextbookAgent."""
        self.config = config or {}
        self.faiss_index_path = self.config.get("faiss_index_path", "data/index")
        self.chunks_file_path = self.config.get("chunks_file_path", "data/chunks.txt")
        try:
            self.textbook_agent = TextbookAgent(config=self.config,
                                                faiss_index_path=self.faiss_index_path,
                                                chunks_file_path=self.chunks_file_path)
        except (FaissIndexError, ChunksFileError) as e:
            logger.error(f"Error initializing TextbookAgent: {e}")
            raise RecommendationAgentError(f"Error initializing TextbookAgent: {e}") from e
        self.max_new_tokens = self.config.get("max_new_tokens", 500)
        self.max_relevant_chunks = self.config.get("max_relevant_chunks", 5)
        self.num_recommendations = self.config.get("num_recommendations", 3)
        self.max_retries = self.config.get("max_retries", 3)
        self.retry_delay = self.config.get("retry_delay", 2)

    def get_resource_recommendations(self, query: str) -> Optional[List[str]]:
        """
        Generates resource recommendations based on a query with retry mechanism for LLM calls.

        Args:
            query (str): The query to generate recommendations from.

        Returns:
            Optional[List[str]]: A list of resource recommendations, or None if an error persists after retries.

        Raises:
            RecommendationAgentError: For errors during recommendation generation that persist after retries.
        """
        try:
            relevant_chunks: List[str] = self.textbook_agent.get_relevant_chunks(
                query, top_k=self.max_relevant_chunks)
        except Exception as e:
            logger.error(f"Error getting relevant chunks: {e}")
            raise RecommendationAgentError(f"Error getting relevant chunks: {e}") from e

        if not relevant_chunks:
            logger.warning("No relevant chunks found.")
            return ["I could not find relevant information in the textbook to generate recommendations."]

        prompt: str = self._build_prompt(query, relevant_chunks)
        logger.debug(f"Generated Prompt: {prompt}")

        for attempt in range(self.max_retries):
            try:
                responses: Optional[List[str]] = self.llm_wrapper.generate_response(
                    prompt,
                    generation_kwargs={"max_new_tokens": self.max_new_tokens,
                                         "temperature": 0.7}
                )
                if responses:
                    recommendations = self._extract_recommendations(responses[0])
                    if recommendations:
                        return recommendations
                    else:
                        logger.error("Failed to extract recommendations from LLM response.")
                        raise RecommendationAgentError("Failed to extract recommendations from LLM response.")
                else:
                    logger.warning("LLM returned an empty response.")
                    return ["No response generated."]
            except LLMConnectionError as e:
                logger.error(f"Attempt {attempt + 1}/{self.max_retries} failed to get recommendations: {e}")
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay ** (attempt + 1))  # Exponential backoff
                else:
                    logger.error(f"All {self.max_retries} attempts to get recommendations failed due to LLM connection issues.")
                    raise RecommendationAgentError(f"Failed to get recommendations after multiple retries due to LLM connection error: {e}") from e
            except RecommendationAgentError as e:
                # Re-raise RecommendationAgentError immediately, as it's likely a data processing issue
                raise
            except Exception as e:
                logger.error(f"An unexpected error occurred during recommendation generation (attempt {attempt + 1}): {e}")
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay ** (attempt + 1))  # Exponential backoff
                else:
                    logger.error(f"All {self.max_retries} attempts to get recommendations failed due to an unexpected error.")
                    raise RecommendationAgentError(f"Failed to get recommendations after multiple retries due to: {e}") from e

        return ["Failed to get recommendations after multiple retries."]  # Should not reach here

    def _build_prompt(self, query: str, context: Optional[List[str]] = None) -> str:
        """
        Builds the prompt for the LLM to recommend learning resources.

        Args:
            query (str): The initial query.
            context (Optional[List[str]]): Relevant text chunks from the textbook.

        Returns:
            str: The formatted prompt.
        """
        prompt = "You are a helpful tutor. Based on the user's query and the following text from a textbook, suggest 3 relevant learning resources (e.g., further readings, websites, videos, interactive tools) that the user might find helpful to learn more about this topic.\n\n"
        if context:
            prompt += "Textbook Excerpts:\n" + "\n\n".join(context) + "\n\n"
        prompt += f"User's Query: {query}\n\n"
        prompt += "Recommended Resources:\n"
        return prompt

    def _extract_recommendations(self, response: str) -> Optional[List[str]]:
        """
        Extracts the list of recommendations from the LLM's response.

        Args:
            response (str): The LLM's response string.

        Returns:
            Optional[List[str]]: A list of recommendations, or None if extraction fails.
        """
        logger.debug(f"Raw LLM Response for Recommendation: {response}")
        lines = response.strip().split('\n')
        recommendations = [line.lstrip('- ').strip() for line in lines if line.strip()]  # Remove leading dashes and whitespace
        if recommendations:
            return recommendations
        else:
            return None


if __name__ == '__main__':
    faiss_path = "data/index"
    llm_config = "config/llm_config.yaml"
    chunks_file_path = "data/chunks.txt"

    # Create dummy config if it doesn't exist
    if not os.path.exists("config"):
        os.makedirs("config")
    if not os.path.exists(llm_config):
        with open(llm_config, 'w') as f:
            yaml.dump({"model_name": "distilgpt2"},
                      f)
        logger.info(f"Created a dummy LLM config at {llm_config}. Please adjust the model_name if needed.")

    # Initialize and use the agent
    try:
        # You would typically get these instances from your main application setup
        class MockLLMWrapper:
            def generate_response(self, prompt, generation_kwargs):
                return ["- Resource 1\n- Resource 2\n- Resource 3"]

        class MockVectorMemory:
            pass

        mock_llm_wrapper = MockLLMWrapper()
        mock_vector_memory = MockVectorMemory()

        agent = RecommendationAgent(config={"llm_wrapper": mock_llm_wrapper, "vector_memory": mock_vector_memory,
                                           "faiss_index_path": faiss_path, "chunks_file_path": chunks_file_path})
        query = "Newton's laws of motion"
        recommendations = agent.get_resource_recommendations(query)
        if recommendations:
            logger.info(f"Recommended resources for '{query}':")
            for rec in recommendations:
                logger.info(f"- {rec}")
        else:
            logger.info(f"Could not get recommendations for '{query}'.")
    except RecommendationAgentError as e:
        logger.error(f"Error: {e}")