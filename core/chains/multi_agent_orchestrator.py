import logging
from typing import Dict, Any, List, Optional
from core.llm_wrapper import LLMWrapper
from core.agents.tutor_agent import TutorAgent
from core.agents.quiz_agent import QuizAgent
from core.agents.recommendation_agent import RecommendationAgent
from core.agents.progress_agent import ProgressAgent

logger = logging.getLogger(__name__)

class AgentManager:
    def __init__(self, config: Dict[str, Any], agents: Optional[Dict[str, Any]] = None):
        """
        Initializes the AgentManager with a configuration and an optional dictionary of agents.
        """
        self.config = config
        self.llm_wrapper: LLMWrapper = config.get("llm_wrapper")
        if not self.llm_wrapper:
            raise ValueError("LLM wrapper instance must be provided in the config.")
        self.vector_memory = config.get("vector_memory")
        if not self.vector_memory:
            raise ValueError("Vector memory instance must be provided in the config.")
        self.agents: Dict[str, Any] = agents if agents is not None else self._initialize_agents()
        logger.info("AgentManager initialized.")

    def _initialize_agents(self) -> Dict[str, Any]:
        """
        Initializes the individual agents managed by this AgentManager.
        """
        llm_config_path = self.config.get("llm_config_path")
        if not llm_config_path:
            raise ValueError("LLM config path must be provided in the config for agent initialization.")

        tutor_agent = TutorAgent(config=self.config)
        logger.info("TutorAgent initialized.")

        quiz_agent = QuizAgent(config=self.config)
        logger.info("QuizAgent initialized.")

        recommendation_agent = RecommendationAgent(config=self.config)
        logger.info("RecommendationAgent initialized.")

        progress_agent = ProgressAgent(config=self.config) # Corrected initialization: Passing the entire config
        logger.info("ProgressAgent initialized.")

        initialized_agents = {
            "tutor": tutor_agent,
            "quiz": quiz_agent,
            "recommendation": recommendation_agent,
            "progress": progress_agent,
        }
        print(f"Initialized agents: {initialized_agents}") # Added debug print
        return initialized_agents

    async def handle_input(self, user_input: str, user_history: List[str], progress: Optional[ProgressAgent] = None) -> Optional[Dict[str, Any]]:
        """
        Routes the user input to the appropriate agent based on the context.
        """
        logger.info(f"AgentManager: Handling user input: '{user_input}'")

        if user_input.lower().startswith("quiz me"):
            logger.info("AgentManager: Routing to QuizAgent.")
            return await self.agents["quiz"].handle_input(user_input, user_history, progress=progress)
        elif user_input.lower().startswith("what should i learn next") or user_input.lower().startswith("recommend a topic"):
            logger.info("AgentManager: Routing to RecommendationAgent.")
            return await self.agents["recommendation"].handle_input(user_input, user_history, progress=progress)
        elif user_input.lower().startswith("progress"):
            logger.info("AgentManager: Routing to ProgressAgent.")
            if progress:
                return await progress.get_progress_summary() # Assuming this is an async method
            else:
                return {"response": "Progress tracking is not available at the moment.", "source": "AgentManager", "confidence": "low"}
        else:
            logger.info("AgentManager: Routing to TutorAgent.")
            # Default to the tutor agent for learning and explanations
            logger.debug(f"AgentManager: Type of agent_to_use: {type(self.agents.get('tutor'))}") # Added debug line
            return await self.agents["tutor"].handle_input(user_input, user_history, progress=progress)