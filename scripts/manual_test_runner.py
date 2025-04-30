# scripts/manual_test_runner.py
import os
import sys

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.agents.ingestion_agent import IngestionAgent
from core.agents.textbook_agent import TextbookAgent
from core.agents.tutor_agent import TutorAgent
from core.agents.quiz_agent import QuizAgent
from core.agents.progress_agent import ProgressAgent
from core.agents.recommendation_agent import RecommendationAgent
from core.chains.multi_agent_orchestrator import AgentManager
from core.llm_wrapper import LLMWrapper  # Corrected import
from core.config import Config  # Import the Config class

def main():
    print("🧪 Manual Test Runner Started...")

    # 0. Load Configuration
    config = Config()

    # 1. Setup Agents and Dependencies
    try:
        ingestion_agent = IngestionAgent(config=config)
        textbook_agent = TextbookAgent(faiss_index_path="data/index.faiss", config=config)
        tutor_agent = TutorAgent(config=config)
        llm = LLMWrapper(config=config)  # Pass the config object
        quiz_agent = QuizAgent(llm=llm, config=config)
        progress_agent = ProgressAgent(config=config)
        recommend_agent = RecommendationAgent(llm=llm, config=config)
        orchestrator = AgentManager(agents={
            "tutor": tutor_agent,
            "quiz": quiz_agent,
            "recommend": recommend_agent,
            "progress": progress_agent,
        }, config=config)  # Pass the config
    except Exception as e:
        print(f"❌ Error during agent setup: {e}")
        return

    # 2. Ingestion Test
    if not os.path.exists("data/index.faiss"):
        print("📄 Ingestion Test: Running ingestion...")
        try:
            ingestion_agent.extract_text("data/raw/grade 7-general science_ethiofetenacom_d837.pdf")
            ingestion_agent.chunk_text("Sample Text about Newton's Laws")
            ingestion_agent.build_faiss_index(ingestion_agent.embed_chunks(["Sample chunk about gravity"]))
            print("✅ Ingestion successful.")
        except Exception as e:
            print(f"❌ Ingestion failed: {e}")
            return
    else:
        print("📄 Ingestion Test: Skipped (already ingested).")

    # 3. Retrieval Test
    print("\n🔍 Retrieval Test: Fetching relevant chunks...")
    try:
        chunks = textbook_agent.get_relevant_chunks("What is gravity?")
        print("Chunks:", chunks)
        if not chunks:  # Add a check for empty chunks
            print("❌ Retrieval failed: No chunks returned.")
    except Exception as e:
        print(f"❌ Retrieval failed: {e}")
        return

    # 4. Tutor Test
    print("\n🧠 Tutor Test: Asking a question...")
    try:
        answer = tutor_agent.answer_question("Explain Newton's First Law")
        print("Answer:", answer)
        if not answer:
            print("❌ Tutor Agent failed: No answer returned.")
    except Exception as e:
        print(f"❌ Tutor Agent failed: {e}")
        return

    # 5. Quiz Test
    print("\n❓ Quiz Test: Generating quiz...")
    try:
        quiz = quiz_agent.generate_quiz("solar system")
        print("Quiz:", quiz)
        if not quiz:
            print("❌ Quiz Agent failed: No quiz generated.")
    except Exception as e:
        print(f"❌ Quiz Agent failed: {e}")
        return

    # 6. Progress Test
    print("\n📈 Progress Test: Checking user progress...")
    try:
        progress_summary = progress_agent.get_progress_summary()
        print("Progress:", progress_summary["response"])
        if not progress_summary:
            print("❌ Progress Agent Failed: No progress summary")
    except Exception as e:
        print(f"❌ Progress Agent failed: {e}")
        return

    # 7. Recommendation Test
    print("\n💬 Recommendation Test: Suggesting a topic...")
    try:
        recommendation = recommend_agent.recommend_topic(["photosynthesis"], progress_agent)
        print("Recommendation:", recommendation["response"])
        if not recommendation:
            print("❌ Recommendation Agent Failed: No recommendation")
    except Exception as e:
        print(f"❌ Recommendation Agent failed: {e}")
        return

    # 8. Orchestration Test
    print("\n🕹️ Orchestration Test: Routing user input...")
    try:
        orchestrator_response = orchestrator.handle_input("Explain Photosynthesis", progress=progress_agent)
        print("Orchestrator Response:", orchestrator_response["response"])
        if not orchestrator_response:
            print("❌ Orchestrator Failed: No response")
    except Exception as e:
        print(f"❌ Orchestrator Test failed: {e}")
        return

    print("\n✅ All manual tests completed.")


if __name__ == "__main__":
    main()
