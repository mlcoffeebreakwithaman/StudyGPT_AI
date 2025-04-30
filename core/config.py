import os
import yaml
from typing import Dict, Any
import logging

# Initialize logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class Config:
    """
    A class to manage configuration settings for the application.  It loads
    settings from a YAML file and provides access to them as attributes.
    """

    def __init__(self, config_file: str = "config/llm_config.yaml"):
        """
        Initializes the Config object.

        Args:
            config_file (str, optional): The path to the YAML configuration file.
                Defaults to "config/llm_config.yaml".
        """
        self.config_file = config_file
        self.data = self._load_config()

    def _load_config(self) -> Dict[str, Any]:
        """
        Loads the configuration from the YAML file.

        Returns:
            dict: A dictionary containing the configuration settings.

        Raises:
            FileNotFoundError: If the configuration file does not exist.
            YAMLError: If there is an error parsing the YAML file.
        """
        if not os.path.exists(self.config_file):
            error_message = f"Config file not found at: {self.config_file}"
            logging.error(error_message)  # Log the error
            raise FileNotFoundError(error_message)
        try:
            with open(self.config_file, "r", encoding="utf-8") as f:
                config_data = yaml.safe_load(f)
            return config_data
        except yaml.YAMLError as e:
            error_message = f"Error parsing config file: {e}"
            logging.error(error_message) # Log the error
            raise yaml.YAMLError(error_message)

    def get(self, key: str, default: Any = None) -> Any:
        """
        Retrieves a configuration value for a given key.

        Args:
            key (str): The key of the configuration value to retrieve.
            default (Any, optional): A default value to return if the key is not
                found. Defaults to None.

        Returns:
            Any: The configuration value, or the default value if the key is not found.
        """
        return self.data.get(key, default)

    def __getattr__(self, key: str) -> Any:
        """
        Allows accessing configuration values as attributes.  If the key is not
        found, it raises an AttributeError, which is the standard behavior
        when trying to access a non-existent attribute.

        Args:
            key (str): The key of the configuration value to retrieve.

        Returns:
            Any: The configuration value.

        Raises:
            AttributeError: If the key is not found in the configuration.
        """
        if key in self.data:
            return self.data[key]
        else:
            error_message = f"Configuration key '{key}' not found."
            logging.error(error_message) # Log the error
            raise AttributeError(error_message)
