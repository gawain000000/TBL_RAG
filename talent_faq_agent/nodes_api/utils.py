import logging
from pathlib import Path
from typing import Dict, Any, List, TypedDict
from fastapi import HTTPException
import json


class GraphState(TypedDict):
    """
    Represents the state of our graph.

    Attributes:
        keys: A dictionary where each key is a string.
        messages: A list that stores string messages.
        langgraph_path: The nodes path of the langgraph executed.
    """

    workflow_state: Dict[str, Any]
    messages: List[Any]  # Assuming BaseMessage is not defined, using Any for now.
    langgraph_path: List[str]


# Helper functions
def load_json_file(file_path: str) -> Dict[str, Any]:
    """
    Load and return content from a JSON file.

    :param file_path: Path to the JSON file.
    :return: Parsed JSON data as a dictionary.
    :raises RuntimeError: If the file is missing or JSON decoding fails.
    """
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        logging.error(f"File not found: {file_path}")
        raise RuntimeError(f"File not found: {file_path}")
    except json.JSONDecodeError as e:
        logging.error(f"JSON decoding error in file {file_path}: {e}")
        raise RuntimeError(f"Error decoding JSON from {file_path}")


def load_text_file(file_path: str) -> str:
    """
    Load and return text content from a file.

    :param file_path: Path to the text file.
    :return: File content as a string.
    :raises RuntimeError: If the file is missing.
    """
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        logging.error(f"File not found: {file_path}")
        raise RuntimeError(f"File not found: {file_path}")


def find_dir_with_file(path: str | Path, file_name: str) -> str:
    """
    Find a directory (or its parents) containing a specific file.

    :param path: Starting directory path.
    :param file_name: File name to search for.
    :return: The directory containing the file.
    :raises FileNotFoundError: If the file is not found in the directory hierarchy.
    """
    path = Path(path).resolve()
    if (path / file_name).exists():
        return str(path)

    for parent in path.parents:
        if (parent / file_name).exists():
            return str(parent)

    logging.error(f"File '{file_name}' not found in {path} or its parent directories.")
    raise FileNotFoundError(f"File '{file_name}' not found.")


def load_config(file_name: str, search_dir: str) -> Dict[str, Any]:
    """
    Locate and load a JSON configuration file.

    :param file_name: Name of the configuration file to load.
    :param search_dir: Directory to start searching for the file.
    :return: Loaded configuration as a dictionary.
    :raises HTTPException: If the file is not found or cannot be loaded.
    """
    try:
        config_dir = find_dir_with_file(search_dir, file_name)
        config_path = Path(config_dir) / file_name
        return load_json_file(config_path)
    except FileNotFoundError as e:
        raise HTTPException(
            status_code=500, detail=f"Configuration file '{file_name}' not found: {e}"
        ) from None
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error loading configuration file '{file_name}': {e}"
        ) from None


# Main execution
try:
    # Locate project and script directories
    script_dir = find_dir_with_file(__file__, "nodes_api_config.json")
    project_dir = find_dir_with_file(script_dir, "env_config.json")
except FileNotFoundError as e:
    raise HTTPException(
        status_code=500,
        detail=f"Required directories for configuration files not found: {e}",
    ) from None

# Load configuration files
try:
    env_config = load_config("env_config.json", project_dir)
    nodes_api_config = load_config("nodes_api_config.json", script_dir)
except HTTPException as e:
    logging.error(f"Error initializing configuration: {e.detail}")
    raise
