import sys
import os
import yaml
import asyncio
import atexit
import logging
import pdb  # Import the debugger

# Add the project's root directory to the Python path
PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, PROJECT_ROOT)

from core.llm_wrapper import LLMWrapper
from core.agents.tutor_agent import TutorAgent
from core.agents.quiz_agent import QuizAgent
from core.agents.recommendation_agent import RecommendationAgent
from core.agents.progress_agent import ProgressAgent
from core.chains.multi_agent_orchestrator import AgentManager
from core.memory.vector_memory import VectorMemory  # Import VectorMemory
from typing import List

# Configure logging to show DEBUG messages
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(module)s - %(message)s')
logger = logging.getLogger(__name__)

async def conduct_quiz(quiz_data: dict, progress_agent: ProgressAgent):
    """Conducts a quiz with the user and records the score."""
    questions = quiz_data.get("questions", [])
    score = 0
    num_questions = len(questions)

    print("\nStarting the quiz!")
    for i, question_data in enumerate(questions):
        print(f"\nQuestion {i + 1}: {question_data['question']}")
        if 'choices' in question_data:
            for choice_index, choice in enumerate(question_data['choices']):
                print(f"  {chr(ord('A') + choice_index)}. {choice}") # Display with A, B, C, D
            user_answer_letter = input("Your answer (A, B, C, or D): ").upper()
            correct_answer_text = question_data.get('answer')
            correct_answer_letter = None
            if correct_answer_text and 'choices' in question_data:
                try:
                    correct_answer_index = question_data['choices'].index(correct_answer_text)
                    correct_answer_letter = chr(ord('A') + correct_answer_index)
                except ValueError:
                    logger.error(f"Correct answer '{correct_answer_text}' not found in choices for question: {question_data['question']}")

            if user_answer_letter == correct_answer_letter:
                print("Correct!")
                score += 1
            elif correct_answer_letter:
                print(f"Incorrect. The correct answer was {correct_answer_letter} ({correct_answer_text}).")
            else:
                print(f"Incorrect. The correct answer could not be determined.")
        else:
            user_answer = input("Your answer: ")
            correct_answer = question_data.get('answer')
            if user_answer.lower() == correct_answer.lower():
                print("Correct!")
                score += 1
            elif correct_answer:
                print(f"Incorrect. The correct answer was: {correct_answer}")
            else:
                print("Incorrect. The correct answer was not provided.")

    final_score = score / num_questions if num_questions > 0 else 0
    print(f"\nQuiz finished! Your score: {score}/{num_questions} ({final_score:.2%})")
    await progress_agent.record_quiz_score(final_score) # Await the progress recording

async def main():
    try:
        # Create dummy config if it doesn't exist
        if not os.path.exists("config"):
            os.makedirs("config")
        llm_config_path = "config/llm_config.yaml"
        if not os.path.exists(llm_config_path):
            with open(llm_config_path, 'w') as f:
                yaml.dump({"model_name": "distilgpt2"}, f)
            logger.info(f"Created a dummy LLM config at {llm_config_path}. Please adjust the model_name if needed.")

        llm_instance = LLMWrapper(config_path=llm_config_path)
        vector_memory = VectorMemory(embedding_model_name=os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2"))
        index_path = "data/index"
        if vector_memory.vectorstore is None:
            vector_memory.load_data("data/raw")
            vector_memory.save_index(index_path)

        agent_manager = AgentManager(
            config={"llm_wrapper": llm_instance, "llm_config_path": llm_config_path, "vector_memory": vector_memory},
            agents={}  # AgentManager will initialize agents
        )

        user_history: List[str] = []

        print("Welcome to StudyGPT!")

        # Access progress agent through agent_manager
        progress_agent = agent_manager.agents.get("progress")
        if progress_agent:
            # Collect user preferences at the start (this will now load existing if available)
            if progress_agent.progress.get("interests") is None:
                interests = input("What are your primary areas of learning interest (e.g., science, history, programming)? ").lower()
                await progress_agent.record_preference("interests", interests) # Await preference recording

            if progress_agent.progress.get("learning_style") is None:
                learning_style = input("Optional: Do you have a preferred learning style (e.g., visual, hands-on, theoretical)? ").lower()
                if learning_style:
                    await progress_agent.record_preference("learning_style", learning_style) # Await preference recording

            if progress_agent.progress.get("knowledge_level") is None:
                knowledge_level = input("Optional: What is your current knowledge level in general (e.g., beginner, intermediate, advanced)? ").lower()
                if knowledge_level:
                    await progress_agent.record_preference("knowledge_level", knowledge_level) # Await preference recording

            # Ensure progress is saved on exit
            atexit.register(progress_agent._save_progress)
        else:
            logger.error("Progress agent not initialized.")

        while True:
            user_input = input("You: ")
            if user_input.lower() in ["quit", "exit", "bye"]:
                print("Goodbye!")
                break

            response = await agent_manager.handle_input(user_input, user_history, progress=progress_agent)
            pdb.set_trace()  # Add this line for debugging
            if response is None:
                logger.error(f"agent_manager.handle_input returned None for input: '{user_input}'")
                print("StudyGPT: Sorry, I encountered an issue processing your request.")
                continue # Skip the rest of the loop and ask for new input

            if "response" in response:
                print("StudyGPT:", await response["response"]) # Await the async response
                if "what should i learn next" in user_input.lower() or "recommend a topic" in user_input.lower():
                    while True:
                        try:
                            rating = int(input("Rate this recommendation (1-5, 5 being most helpful): "))
                            if 1 <= rating <= 5:
                                break
                            else:
                                print("Rating must be between 1 and 5.")
                        except ValueError:
                            print("Invalid input. Please enter a number between 1 and 5.")
                    reason = input("Optional: Why did you choose this rating? ")
                    if progress_agent and "response" in response:
                        await progress_agent.record_recommendation_feedback(response["response"], rating, reason) # Await feedback recording
            elif "questions" in response: # Handle direct quiz response
                await conduct_quiz(response, progress_agent) # Await the async quiz
            else:
                print("StudyGPT:", response) # Print raw response if 'response' key is missing

            user_history.append(user_input)

    except Exception as e:
        logging.error(f"An error occurred in the main AI loop: {e}")
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    asyncio.run(main())