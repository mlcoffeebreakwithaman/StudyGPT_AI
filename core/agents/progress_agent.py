import logging
import json
from typing import Optional, Dict, List, Any
from core.exceptions import ProgressAgentError  # Import

# Initialize logging for this module
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(name)s - %(message)s')
logger = logging.getLogger(__name__)

class ProgressAgent:
    def __init__(self, config: Optional[Dict[str, Any]] = None, **kwargs: Any):
        """
        Initializes the ProgressAgent.
        """
        super().__init__()  # Corrected line: Calling super().__init__() without arguments
        if config:
            self.vector_memory = config.get("vector_memory")
        else:
            raise ValueError("Configuration dictionary must be provided to ProgressAgent.")
        self.config = config or {}
        self.save_file = self.config.get("progress_save_file", "progress.json")
        self.progress = self._load_progress()
        logger.info("ProgressAgent initialized.")

    def _load_progress(self) -> Dict:
        """
        Loads progress data from the save file, if it exists, and validates its structure.
        """
        default_progress = {
            "interests": None,
            "learning_style": None,
            "knowledge_level": None,
            "topics_covered": [],
            "quizzes_taken": 0,
            "quiz_scores": [],
            "questions_asked": 0,
            "recommendation_feedback": [],
        }
        try:
            with open(self.save_file, "r") as f:
                loaded_progress = json.load(f)
                # Basic validation of loaded data structure
                if not isinstance(loaded_progress, dict) or set(default_progress.keys()) != set(loaded_progress.keys()):
                    logger.error("ProgressAgent: Save file has invalid structure, initializing new progress.")
                    return default_progress
                # Further type checking can be added here if needed
                return loaded_progress
        except FileNotFoundError:
            logger.info("ProgressAgent: Save file not found, initializing new progress.")
            return default_progress
        except json.JSONDecodeError:
            logger.error("ProgressAgent: Error decoding save file, initializing new progress.")
            return default_progress

    def _save_progress(self):
        """
        Saves the current progress data to the save file.
        """
        try:
            with open(self.save_file, "w") as f:
                json.dump(self.progress, f, indent=4)
            logger.info("ProgressAgent: Progress saved.")
        except Exception as e:
            logger.error(f"ProgressAgent: Error saving progress: {e}")
            raise ProgressAgentError(f"Error saving progress: {e}")  # Raise

    def record_preference(self, preference_type: str, value: str):
        """
        Records the user's learning preferences.
        """
        if preference_type not in self.progress:  # check
            logger.error(f"Invalid preference type: {preference_type}")
            raise ProgressAgentError(f"Invalid preference type: {preference_type}")
        self.progress[preference_type] = value
        logger.info(f"ProgressAgent: Recorded preference - {preference_type}: {value}")
        self._save_progress()

    def record_topic_covered(self, topic: str):
        """
        Records that the user has covered a specific topic.
        """
        if topic not in self.progress["topics_covered"]:
            self.progress["topics_covered"].append(topic)
            logger.info(f"ProgressAgent: Recorded topic covered - {topic}")
            self._save_progress()

    def record_quiz_taken(self):
        """
        Increments the count of quizzes taken.
        """
        self.progress["quizzes_taken"] += 1
        logger.info("ProgressAgent: Recorded quiz taken.")
        self._save_progress()

    def record_quiz_score(self, score: float):
        """
        Records the score of a quiz taken.
        """
        self.progress["quiz_scores"].append(score)
        logger.info(f"ProgressAgent: Recorded quiz score - {score}")
        self._save_progress()

    def record_question_asked(self, question: str):
        """
        Records a question asked by the user.
        """
        self.progress["questions_asked"] += 1
        logger.info(f"ProgressAgent: Recorded question asked - {question}")
        self._save_progress()

    def record_recommendation_feedback(self, recommendation: str, rating: int, reason: Optional[str] = None):
        """
        Records the user's feedback on a recommendation.
        """
        feedback_entry = {"recommendation": recommendation, "rating": rating}
        if reason:
            feedback_entry["reason"] = reason
        self.progress["recommendation_feedback"].append(feedback_entry)
        logger.info(f"ProgressAgent: Recorded feedback on '{recommendation}' - Rating: {rating}, Reason: {reason}")
        self._save_progress()

    def get_progress_summary(self) -> Dict:
        """
        Returns a summary of the user's progress.
        """
        summary_text = "Your Learning Progress:\n"
        if self.progress["interests"]:
            summary_text += f"- Interests: {self.progress['interests']}\n"
        if self.progress["learning_style"]:
            summary_text += f"- Preferred Learning Style: {self.progress['learning_style']}\n"
        if self.progress["knowledge_level"]:
            summary_text += f"- Current Knowledge Level: {self.progress['knowledge_level']}\n"
        summary_text += f"- Topics Covered: {len(self.progress['topics_covered'])}\n"
        summary_text += f"- Quizzes Taken: {self.progress['quizzes_taken']}\n"
        if self.progress["quiz_scores"]:
            avg_score = sum(self.progress["quiz_scores"]) / len(self.progress["quiz_scores"])
            summary_text += f"- Average Quiz Score: {avg_score:.2%}\n"
            last_scores = self.progress["quiz_scores"][-3:]
            if last_scores:
                summary_text += f"- Last Quiz Scores: {', '.join([f'{s:.2%}' for s in last_scores])}\n"
        summary_text += f"- Questions Asked: {self.progress['questions_asked']}\n"
        summary_text += f"- Recommendation Feedback: {len(self.progress['recommendation_feedback'])} entries\n"
        if self.progress["recommendation_feedback"]:
            last_feedback = self.progress["recommendation_feedback"][-1]
            summary_text += f"    (Last feedback: '{last_feedback['recommendation']}' - Rating: {last_feedback['rating']}"
            if "reason" in last_feedback:
                summary_text += f", Reason: {last_feedback['reason']}"
            summary_text += ")\n"

        return {"response": summary_text, "source": "ProgressAgent", "confidence": "high"}

# Example usage (for testing within this file)
if __name__ == "__main__":
    # Mock VectorMemory for standalone testing
    class MockVectorMemory:
        pass

    mock_vector_memory = MockVectorMemory()

    progress_agent = ProgressAgent(config={"vector_memory": mock_vector_memory})
    progress_agent.record_preference("interests", "physics, astronomy")
    progress_agent.record_preference("learning_style", "visual")
    progress_agent.record_topic_covered("Newton's Laws")
    progress_agent.record_quiz_score(0.9)
    progress_agent.record_recommendation_feedback("Relativity", 3, "A bit confusing.")
    logger.info(f"\n\u23f0 Progress Summary: {progress_agent.get_progress_summary()['response']}")

    try:
        progress_agent.record_preference("invalid_preference", "blah")
    except ProgressAgentError as e:
        logger.error(f"Error recording preference: {e}")