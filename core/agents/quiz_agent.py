# QuizAgent.py
import sys
import os
import yaml
import random
from typing import List, Optional, Dict, Tuple
import logging
from collections import deque
import time
import json # Import json

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(name)s - %(message)s')
logger = logging.getLogger(__name__)

# Add the project root directory to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, project_root)

from core.agents.textbook_agent import TextbookAgent, FaissIndexError, ChunksFileError
from core.llm_wrapper import LLMWrapper, LLMConnectionError, AgentConfigError
from core.exceptions import QuizAgentError # Import

class QuizAgent:
    def __init__(self, llm_wrapper: LLMWrapper, vector_memory, config: Optional[Dict] = None,
                 faiss_index_path: str = "data/index",  # Changed default to the directory
                 llm_config_path: str = "config/llm_config.yaml",
                 chunks_file_path: str = "data/chunks.txt"):
        """
        Initializes the QuizAgent.

        Args:
            llm_wrapper (LLMWrapper): An instance of the LLMWrapper.
            vector_memory: An instance of the VectorMemory.
            config (Optional[Dict]): Configuration dictionary. Defaults to None.
            faiss_index_path (str): Path to the FAISS index directory. # Updated description
            llm_config_path (str): Path to the LLM configuration file.
            chunks_file_path (str): Path to the text chunks file.
        """
        self.llm_wrapper = llm_wrapper
        self.vector_memory = vector_memory
        self.config = config or {}
        self.faiss_index_path = faiss_index_path
        self.chunks_file_path = chunks_file_path
        try:
            self.textbook_agent = TextbookAgent(config=config, faiss_index_path=faiss_index_path,
                                                chunks_file_path=chunks_file_path)
        except (FaissIndexError, ChunksFileError) as e:
            logger.error(f"Error initializing TextbookAgent: {e}")
            raise QuizAgentError(f"Error initializing TextbookAgent: {e}") from e
        self.max_new_tokens = 700 # Increased for more verbose output
        self.max_relevant_chunks = 3
        self.quiz_history = deque(maxlen=5)  # Keep a history of the last 5 quizzes
        self.max_retries = 3
        self.retry_delay = 2  # seconds

    def generate_quiz(self, query: str) -> Optional[Dict]:
        """
        Generates a quiz based on a query with retry mechanism for LLM calls.

        Args:
            query (str): The query to generate the quiz from.

        Returns:
            Optional[Dict]: A dictionary containing the quiz questions and answers,
                            or None if an error occurs after retries.
        Raises:
            QuizAgentError: For errors during quiz generation that persist after retries.
        """
        try:
            relevant_chunks: List[str] = self.textbook_agent.get_relevant_chunks(
                query, top_k=self.max_relevant_chunks)
        except Exception as e:
            logger.error(f"Error getting relevant chunks: {e}")
            raise QuizAgentError(f"Error getting relevant chunks: {e}") from e

        if not relevant_chunks:
            logger.warning("No relevant chunks found.")
            return {"error": "I could not find relevant information in the textbook to generate a quiz."}

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
                    quiz_data = self._extract_quiz_data(responses[0])
                    if quiz_data:
                        self.quiz_history.append(quiz_data)  # Store the quiz
                        return quiz_data
                    else:
                        logger.error("Failed to extract quiz data from LLM response.")
                        raise QuizAgentError("Failed to extract quiz data from LLM response.")
                else:
                    logger.warning("LLM returned an empty response.")
                    return {"error": "No response generated."}
            except LLMConnectionError as e:
                logger.error(f"Attempt {attempt + 1}/{self.max_retries} failed to generate quiz: {e}")
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay ** (attempt + 1))  # Exponential backoff
                else:
                    logger.error(f"All {self.max_retries} attempts to generate quiz failed due to LLM connection issues.")
                    raise QuizAgentError(f"Failed to generate quiz after multiple retries due to LLM connection error: {e}") from e
            except QuizAgentError as e:
                # Re-raise QuizAgentError immediately, as it's likely a data processing issue
                raise
            except Exception as e:
                logger.error(f"An unexpected error occurred during quiz generation (attempt {attempt + 1}): {e}")
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay ** (attempt + 1))  # Exponential backoff
                else:
                    logger.error(f"All {self.max_retries} attempts to generate quiz failed due to an unexpected error.")
                    raise QuizAgentError(f"Failed to generate quiz after multiple retries due to: {e}") from e

        return None # Should not reach here if exceptions are correctly handled/re-raised

    def _build_prompt(self, query: str, context: Optional[List[str]] = None) -> str:
        """
        Builds the prompt for the LLM to generate a quiz with multiple-choice questions.

        Args:
            query (str): The query string.
            context (Optional[List[str]]): A list of relevant text chunks.

        Returns:
            str: The formatted prompt.
        """
        prompt = "You are a helpful tutor. Generate a quiz with 1 multiple-choice question based on the following text. " # Reduced to 1 for testing
        prompt += "Each question should have 4 options labeled A, B, C, D. Clearly indicate the correct answer.\n\n"
        if context:
            prompt += "Context:\n" + "\n\n".join(context) + "\n\n"
        prompt += f"Query: {query}\n\n"
        prompt += "Format the output as a JSON object with a single key 'questions' which is a list of dictionaries. "
        prompt += "Each dictionary should have the keys 'question' (the question text), 'choices' (a list of four option strings), and 'answer' (the correct answer as a string - the text of the correct choice).\n\n"
        prompt += "For example:\n"
        prompt += "{\n"
        prompt += "  'questions': [\n"
        prompt += "    {\n"
        prompt += "      'question': 'What is the capital of France?',\n"
        prompt += "      'choices': ['Berlin', 'Paris', 'Madrid', 'Rome'],\n"
        prompt += "      'answer': 'Paris'\n"
        prompt += "    }\n"
        prompt += "  ]\n"
        prompt += "}\n"
        return prompt

    def _extract_quiz_data(self, response: str) -> Optional[Dict]:
        """
        Extracts quiz questions and answers (with choices) from the LLM response.

        Args:
            response (str): The LLM's response string.

        Returns:
            Optional[Dict]: A dictionary containing the quiz questions (with choices and answers),
                            or None if extraction fails.
        """
        logger.debug(f"Raw LLM Response for Quiz: {response}") # Added debug log
        try:
            quiz_data = json.loads(response)
            if not isinstance(quiz_data, dict) or 'questions' not in quiz_data or not isinstance(quiz_data['questions'], list):
                logger.error(f"LLM response format is incorrect: {response}")
                return None
            for q_data in quiz_data['questions']:
                if not isinstance(q_data, dict) or \
                   'question' not in q_data or not isinstance(q_data['question'], str) or \
                   'choices' not in q_data or not isinstance(q_data['choices'], list) or len(q_data['choices']) != 4 or \
                   'answer' not in q_data or not isinstance(q_data['answer'], str) or q_data['answer'] not in q_data['choices']:
                    logger.error(f"Invalid question format in LLM response: {q_data}")
                    return None
            return quiz_data
        except json.JSONDecodeError:
            logger.error(f"LLM response was not valid JSON: {response}")
            return None
        except Exception as e:
            raise QuizAgentError(f"Error extracting quiz data: {e}. Response: {response}") from e

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
        class MockLLMWrapper:
            def generate_response(self, prompt, generation_kwargs):
                return ["""
                {
                  "questions": [
                    {
                      "question": "What is the powerhouse of the cell?",
                      "choices": ["Nucleus", "Mitochondria", "Ribosome", "Endoplasmic Reticulum"],
                      "answer": "Mitochondria"
                    }
                  ]
                }
                """]

        class MockVectorMemory:
            pass

        mock_llm_wrapper = MockLLMWrapper()
        mock_vector_memory = MockVectorMemory()

        agent = QuizAgent(llm_wrapper=mock_llm_wrapper, vector_memory=mock_vector_memory, llm_config_path=llm_config, faiss_index_path=faiss_path, chunks_file_path=chunks_file_path) # Pass the directory
        query = "Cell biology"
        quiz = agent.generate_quiz(query)
        if quiz and "error" not in quiz:
            logger.info(f"Quiz on: {query}")
            for q_data in quiz['questions']:
                logger.info(f"\nQuestion: {q_data['question']}")
                for i, choice in enumerate(q_data['choices']):
                    logger.info(f"  {chr(ord('A') + i)}. {choice}")
                logger.info(f"Correct Answer: {q_data['answer']}\n")
        elif quiz:
            logger.error(f"Error: {quiz['error']}")
        else:
            logger.error("Failed to generate quiz.")
    except QuizAgentError as e:
        logger.error(f"Error: {e}")