import logging
from typing import Dict, List, Union
from core.agents.base_agent import BaseAgent
from core.agents.progress_agent import ProgressAgent

logger = logging.getLogger(__name__)

class AgentManager:
    def __init__(self, config: dict, agents: Dict[str, BaseAgent]):
        self.config = config
        self.llm_wrapper = config.get("llm_wrapper")
        self.vector_memory = config.get("vector_memory")
        self.agents: Dict[str, BaseAgent] = self._initialize_agents(agents)
        logger.info("All agents initialized.")

    def _initialize_agents(self, initial_agents: Dict[str, BaseAgent]) -> Dict[str, BaseAgent]:
        agents = {}
        llm_config = {"llm_wrapper": self.llm_wrapper, "vector_memory": self.vector_memory}
        # Initialize Tutor Agent
        if "tutor" not in initial_agents:
            from core.agents.tutor_agent import TutorAgent
            agents["tutor"] = TutorAgent(config=llm_config)
            logger.info("TutorAgent initialized.")
        else:
            agents["tutor"] = initial_agents["tutor"]

        # Initialize Quiz Agent
        if "quiz" not in initial_agents:
            from core.agents.quiz_agent import QuizAgent
            agents["quiz"] = QuizAgent(config=llm_config)
            logger.info("QuizAgent initialized.")
        else:
            agents["quiz"] = initial_agents["quiz"]

        # Initialize Recommendation Agent
        if "recommendation" not in initial_agents:
            from core.agents.recommendation_agent import RecommendationAgent
            agents["recommendation"] = RecommendationAgent(config=llm_config)
            logger.info("RecommendationAgent initialized.")
        else:
            agents["recommendation"] = initial_agents["recommendation"]

        # Initialize Progress Agent
        if "progress" not in initial_agents:
            from core.agents.progress_agent import ProgressAgent
            agents["progress"] = ProgressAgent(config={}) # Progress agent doesn't need LLM or vector memory
            logger.info("ProgressAgent initialized.")
        else:
            agents["progress"] = initial_agents["progress"]

        return agents

    async def _determine_relevant_agent(self, user_input: str) -> Union[str, None]:
        # Simple keyword-based routing for now
        if "quiz" in user_input.lower() or "test" in user_input.lower() or "assess" in user_input.lower():
            return "quiz"
        elif "recommend" in user_input.lower() or "suggest" in user_input.lower() or "what should i learn" in user_input.lower():
            return "recommendation"
        elif "progress" in user_input.lower() or "how am i doing" in user_input.lower() or "my score" in user_input.lower():
            return "progress"
        else:
            return "tutor" # Default to tutor agent

    async def handle_input(self, user_input: str, user_history: List[str] = None, progress: ProgressAgent = None) -> Union[dict, str, None]:
        logger.info(f"AgentManager: Handling input: '{user_input}'")
        if progress:
            await progress.record_question_asked(user_input)

        # Determine the most relevant agent
        relevant_agent_name = await self._determine_relevant_agent(user_input)
        logger.info(f"AgentManager: Relevant agent determined: '{relevant_agent_name}'")

        if relevant_agent_name:
            agent = self.agents.get(relevant_agent_name)
            if agent:
                logger.info(f"AgentManager: Calling agent: '{relevant_agent_name}'")
                try:
                    response = await agent.handle_input(user_input, user_history=user_history, progress=progress)
                    logger.info(f"AgentManager: Response received from '{relevant_agent_name}': {response}") # Log the response
                    return response
                except Exception as e:
                    logger.error(f"AgentManager: Error calling agent '{relevant_agent_name}': {e}")
                    return None
            else:
                logger.warning(f"AgentManager: Relevant agent '{relevant_agent_name}' not found.")
                return None
        else:
            logger.warning("AgentManager: No relevant agent found for the input.")
            return None