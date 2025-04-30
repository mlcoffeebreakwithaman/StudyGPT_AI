import transformers
import torch
import logging
import yaml
import os
from typing import List, Optional, Dict, Any
import time

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(name)s - %(message)s"
)
logger = logging.getLogger(__name__)


# Define custom exceptions
class AgentConfigError(Exception):
    """
    Exception raised for errors in agent configuration.
    """
    def __init__(self, message: str, config_path: Optional[str] = None):
        super().__init__(message)
        self.config_path = config_path
        self.message = message

class LLMConnectionError(Exception):
    """
    Exception raised when there is an error connecting to or communicating
    with the language model (LLM).
    """
    def __init__(self, message: str, details: Optional[Dict] = None):
        super().__init__(message)
        self.details = details
        self.message = message


class LLMWrapper:
    """
    A wrapper class for interacting with a pre-trained language model (LLM)
    from Hugging Face Transformers. Handles model loading, configuration,
    and text generation.
    """

    def __init__(self, config: Optional[Dict] = None, config_path: str = "config/llm_config.yaml"):
        """
        Initializes the LLMWrapper.

        Args:
            config (Optional[Dict]): Optional configuration dictionary. If
                provided, overrides values from the config file.
            config_path: Path to the YAML configuration file. Defaults to
                "config/llm_config.yaml".
        """
        self.config_path = config_path
        self.raw_config = self._load_config(config_path)  # Store the raw config
        self.config = config or self.raw_config # Override with passed config
        self.model_name = self.config.get(
            "model_name", "TheBloke/TinyLlama-1.1B-Chat"
        )
        self.device = self._get_device()
        self.tokenizer = None
        self.model = None
        self.max_model_length = 2048  # Default, updated in load_model
        self._load_model()  # Use the improved loading method
        self.max_retries = 3
        self.retry_delay = 2  # seconds

    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """
        Loads the LLM configuration from a YAML file.

        Args:
            config_path: Path to the YAML configuration file.

        Returns:
            A dictionary containing the configuration.

        Raises:
            AgentConfigError: If the configuration file does not exist or
                there is an error parsing it.
        """
        if not os.path.exists(config_path):
            raise AgentConfigError(f"LLM config file not found at {config_path}", config_path=config_path)
        try:
            with open(config_path, "r") as f:
                config = yaml.safe_load(f)
            return config
        except yaml.YAMLError as e:
            raise AgentConfigError(f"Error parsing YAML config: {e}", config_path=config_path)

    def _get_device(self) -> torch.device:
        """
        Determines the appropriate device (CPU or GPU) for the model.

        Returns:
            A torch.device object representing the selected device.
        """
        if torch.cuda.is_available():
            device_name = torch.cuda.get_device_name(0)
            logger.info(f"Using PyTorch device: {device_name}")
            return torch.device("cuda")
        else:
            logger.info("Using PyTorch device: cpu")
            return torch.device("cpu")

    def _load_model(self) -> None:
        """
        Loads the pre-trained language model and its tokenizer.
        Handles potential errors during loading and sets the
        `max_model_length` attribute.
        """
        logger.info(f"Loading model: {self.model_name} on device: {self.device}")
        try:
            self.tokenizer = transformers.AutoTokenizer.from_pretrained(
                self.model_name
            )
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                logger.info(f"Forced padding token to: {self.tokenizer.pad_token}")

            logger.info(f"Tokenizer EOS token: {self.tokenizer.eos_token}")
            logger.info(f"Tokenizer PAD token: {self.tokenizer.pad_token}")

            self.model = (
                transformers.AutoModelForCausalLM.from_pretrained(
                    self.model_name
                )
                .to(self.device)
            )

            # Determine max_model_length
            if hasattr(self.model.config, "max_position_embeddings"):
                self.max_model_length = (
                    self.model.config.max_position_embeddings
                )
                logger.info(
                    f"Set max_model_length to {self.max_model_length} "
                    "from model config"
                )
            elif hasattr(self.tokenizer, "model_max_length"):
                self.max_model_length = self.tokenizer.model_max_length
                logger.info(
                    f"Set max_model_length to {self.max_model_length} "
                    "from tokenizer config."
                )
            else:
                self.max_model_length = 2048
                logger.warning(
                    "Could not determine max_model_length. "
                    f"Using default of {self.max_model_length}. "
                    "This may cause errors."
                )
            self.model.eval()  # Set model to evaluation mode
            logger.info(f"Model {self.model_name} loaded successfully.")
        except Exception as e:
            # Wrap any exception during model loading as LLMConnectionError
            raise LLMConnectionError(f"Error loading model: {e}", details={"model_name": self.model_name}) from e

    def generate_response(
        self, prompt: str, context: str, generation_kwargs: Optional[Dict[str, Any]] = None
    ) -> Optional[List[str]]:
        """
        Generates a response from the language model with retry mechanism,
        now explicitly taking context as an argument.

        Args:
            prompt: The input question string.
            context: The relevant context string to provide to the model.
            generation_kwargs: Optional dictionary of keyword arguments
                to pass to the model's `generate` method. See the
                Hugging Face Transformers documentation for available
                arguments. If not provided, default values from the
                configuration are used.

        Returns:
            A list containing the generated response string, or None
            if an error persists after retries.
        """
        if self.model is None or self.tokenizer is None:
            logger.error("Model not loaded. Cannot generate response.")
            return None

        for attempt in range(self.max_retries):
            try:
                formatted_prompt = f"Context: {context}\n\nQuestion: {prompt}\n\nAnswer:"
                inputs = self.tokenizer.encode(
                    formatted_prompt, return_tensors="pt"
                ).to(self.device)

                # Truncate input if it exceeds the model's maximum length
                input_length = inputs.size(1)
                if input_length > self.max_model_length:
                    logger.warning(
                        f"Input prompt ({input_length} tokens) exceeds "
                        f"max_model_length ({self.max_model_length}). Truncating."
                    )
                    inputs = inputs[:, : self.max_model_length]

                # Use values from config if not provided in generation_kwargs
                default_generation_kwargs = {
                    "max_new_tokens": self.config.get("max_new_tokens", 200),
                    "temperature": self.config.get("temperature", 0.7),
                    "do_sample": True,
                    "pad_token_id": self.tokenizer.pad_token_id,
                    "eos_token_id": self.tokenizer.eos_token_id,
                }

                # Override defaults with user-provided kwargs
                final_generation_kwargs = default_generation_kwargs.copy()
                if generation_kwargs:
                    final_generation_kwargs.update(generation_kwargs)

                with torch.no_grad():
                    output = self.model.generate(
                        inputs,
                        attention_mask=torch.ones_like(inputs),  # Add attention mask here
                        **final_generation_kwargs
                    )
                    logger.debug(f"Shape of the output tensor: {output.shape}")
                    logger.debug(f"Content of the output tensor: {output}")
                    response = self.tokenizer.decode(
                        output[0], skip_special_tokens=True
                    ).strip()
                    return [response]
            except LLMConnectionError as e:
                logger.error(f"Attempt {attempt + 1}/{self.max_retries} failed to generate response: {e}")
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay ** (attempt + 1))  # Exponential backoff
                else:
                    logger.error(f"All {self.max_retries} attempts to generate response failed.")
                    raise  # Re-raise the last exception
            except Exception as e:
                logger.error(f"An unexpected error occurred during response generation (attempt {attempt + 1}): {e}")
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay ** (attempt + 1))  # Exponential backoff
                else:
                    logger.error(f"All {self.max_retries} attempts to generate response failed due to unexpected error.")
                    raise LLMConnectionError(f"Failed to generate response after multiple retries due to: {e}", details={"prompt": prompt, "context": context}) from e

        return None # Should not reach here if exceptions are correctly handled/re-raised