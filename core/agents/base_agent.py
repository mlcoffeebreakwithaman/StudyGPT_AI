from typing import Any, Dict
from core.config import Config  # Assuming you have a Config class

class BaseAgent:
    """
    Base class for all agents in the system.
    Provides common functionalities and attributes.
    """

    def __init__(self, config: Config, **kwargs: Any):
        """
        Initializes the base agent with a configuration object.

        Args:
            config (Config): The configuration object for the agent.
            **kwargs (Any): Additional keyword arguments for specific agent needs.
        """
        self.config = config
        self._memory: Dict[str, Any] = {}  # Basic in-memory storage for agents
        # You might add other common attributes here, like an LLM client, etc.
        for key, value in kwargs.items():
            setattr(self, key, value)

    def get_memory(self, key: str) -> Any:
        """
        Retrieves a value from the agent's memory.
        """
        return self._memory.get(key)

    def set_memory(self, key: str, value: Any) -> None:
        """
        Sets a value in the agent's memory.
        """
        self._memory[key] = value

    def clear_memory(self, key: str) -> None:
        """
        Clears a specific key from the agent's memory.
        """
        if key in self._memory:
            del self._memory[key]

    def execute(self, task: str, **kwargs: Any) -> Any:
        """
        Defines the main execution logic for the agent.
        This method should be overridden by subclasses.
        """
        raise NotImplementedError(f"{self.__class__.__name__} must implement the execute method.")

    # You might add other common utility methods here, such as:
    # - Methods for interacting with a common knowledge base.
    # - Logging methods.
    # - Data validation methods.