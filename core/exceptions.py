"""
This module defines custom exception classes for the application.  It's best practice to
keep all custom exceptions in a single module.
"""
from typing import Optional  # Import Optional


class AgentError(Exception):
    """
    Base exception class for agent-related errors.  All other agent-specific
    exceptions should inherit from this class.
    """

    def __init__(self, message: str, agent_name: str = "BaseAgent"):
        """
        Initializes the AgentError.

        Args:
            message (str): The error message.
            agent_name (str, optional): The name of the agent where the error occurred.
                Defaults to "BaseAgent".
        """
        super().__init__(message)
        self.agent_name = agent_name
        self.message = message # Added self.message
        #  Removed logging here.  Logging should be handled where the exception is *caught*,
        #  not where it's defined.  This allows for more flexibility in how errors are handled.

    def __str__(self):
        return f"{self.agent_name} - {self.message}"


class TutorAgentError(AgentError):
    """
    Exception class for errors specific to the TutorAgent.
    """

    def __init__(self, message: str):
        super().__init__(message, agent_name="TutorAgent")


class QuizAgentError(AgentError):
    """
    Exception class for errors specific to the QuizAgent.
    """

    def __init__(self, message: str):
        super().__init__(message, agent_name="QuizAgent")



class RecommendationAgentError(AgentError):
    """
    Exception class for errors specific to the RecommendationAgent.
    """

    def __init__(self, message: str):
        super().__init__(message, agent_name="RecommendationAgent")


class ProgressAgentError(AgentError):
    """
    Exception class for errors specific to the ProgressAgent.
    """

    def __init__(self, message: str):
        super().__init__(message, agent_name="ProgressAgent")


class LLMWrapperError(Exception):
    """
    Base exception for LLMWrapper errors.
    """
    def __init__(self, message: str):
        super().__init__(message)
        self.message = message

class AgentConfigError(LLMWrapperError):  # Changed inheritance
    """
    Exception raised for errors in agent configuration, specifically within LLMWrapper.
    """
    def __init__(self, message: str, config_path: Optional[str] = None):
        super().__init__(message)  # Call LLMWrapperError constructor
        self.config_path = config_path
        self.message = message # added self.message
        # Removed logging.  Handle logging where this exception is caught.

    def __str__(self):
        if self.config_path:
            return f"AgentConfigError: {self.message} Config Path: {self.config_path}"
        else:
            return f"AgentConfigError: {self.message}"



class LLMConnectionError(LLMWrapperError):
    """
    Exception raised for errors related to connecting to or using the LLM.
    """
    def __init__(self, message: str):
        super().__init__(message)
        self.message = message # Added self.message

    def __str__(self):
        return f"LLMConnectionError: {self.message}"

class DataIngestionError(Exception):
    """
    Exception raised for errors during data ingestion.
    """

    def __init__(self, message: str, ingestion_step: Optional[str] = None):
        super().__init__(message)
        self.ingestion_step = ingestion_step
        self.message = message # Added self.message

    def __str__(self):
        if self.ingestion_step:
            return f"DataIngestionError at {self.ingestion_step}: {self.message}"
        return f"DataIngestionError: {self.message}"

class OrchestratorError(Exception):
    """
    Exception raised for errors specific to the Orchestrator.
    """
    def __init__(self, message: str):
        super().__init__(message)
        self.message = message

    def __str__(self):
        return f"OrchestratorError: {self.message}"
