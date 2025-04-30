from StudyGPT_AI.core.llm_wrapper import LLMWrapper
from core.agents.tutor_agent import TutorAgent
import logging

# Initialize logging for the test script with DEBUG level
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(module)s - %(message)s')

if __name__ == "__main__":
    try:
        llm = LLMWrapper()
        tutor = TutorAgent(llm=llm)
        logging.debug(f"TutorAgent initialized: {tutor}")
        logging.debug(f"LLMWrapper instance: {llm}")

        question = "Explain the concept of cognitive dissonance and give a real-world example."
        logging.info(f"Testing tutor.explain with question: '{question}'")
        response = tutor.explain(question)
        logging.debug(f"Response from tutor.explain: {response}")

        print("\n🧠 Tutor says:", response["response"])
        print("\n📚 Source:", response["source"])
        print("🤔 Confidence:", response["confidence"])

        question_with_context = "Explain the role of mitochondria in cellular respiration."
        context = "Cellular respiration is the process by which cells convert glucose into energy in the form of ATP."
        logging.info(f"Testing tutor.explain with question: '{question_with_context}' and context: '{context}'")
        response_with_context = tutor.explain(question_with_context, context=context)
        logging.debug(f"Response from tutor.explain (with context): {response_with_context}")

        print("\n🧠 Tutor says (with context):", response_with_context["response"])
        print("\n📚 Source:", response_with_context["source"])
        print("🤔 Confidence:", response_with_context["confidence"])

    except Exception as e:
        logging.error(f"An error occurred during the tutor test: {e}")