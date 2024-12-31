import json
import yaml
import logging
from pathlib import Path
from fastapi import HTTPException
from typing import Dict, Any, Union, TypedDict, List, AsyncGenerator


def find_dir_with_file(
        path: Union[str, Path],
        name: str,
) -> str:
    """
    Recursively search for a directory (or any parent directory) containing a specific file.

    Args:
        path (Union[str, Path]): Starting directory path for the search.
        name (str): Name of the file to search for.

    Returns:
        str: The path of the directory containing the file.

    Raises:
        FileNotFoundError: If the file is not found in the directory or any of its parents.
    """
    path = Path(path).resolve()
    if (path / name).is_file():
        return str(path)

    for parent in path.parents:
        if (parent / name).is_file():
            return str(parent)

    raise FileNotFoundError(f"File '{name}' not found in '{path}' or its parent directories.")


def load_json_file(file_path: Union[str, Path]) -> Dict[str, Any]:
    """
    Load a JSON file and return its contents as a dictionary.

    Args:
        file_path (Union[str, Path]): Path to the JSON file.

    Returns:
        Dict[str, Any]: The parsed JSON content.

    Raises:
        FileNotFoundError: If the JSON file does not exist.
        json.JSONDecodeError: If the file content is not valid JSON.
    """
    file_path = Path(file_path).resolve()
    if not file_path.is_file():
        raise FileNotFoundError(f"JSON file '{file_path}' not found.")

    with file_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_text_file(file_path: Union[str, Path]) -> str:
    """
    Load a text file and return its content as a string.

    Args:
        file_path (Union[str, Path]): Path to the text file.

    Returns:
        str: The content of the text file.

    Raises:
        FileNotFoundError: If the text file does not exist.
    """
    file_path = Path(file_path).resolve()
    if not file_path.is_file():
        raise FileNotFoundError(f"Text file '{file_path}' not found.")

    with file_path.open("r", encoding="utf-8") as f:
        return f.read()


def load_yaml_file(file_path: Union[str, Path]) -> Dict[str, Any]:
    """
    Load a YAML file and return its contents as a dictionary.

    Args:
        file_path (Union[str, Path]): Path to the YAML file.

    Returns:
        Dict[str, Any]: The parsed YAML content.

    Raises:
        FileNotFoundError: If the YAML file does not exist.
        yaml.YAMLError: If the file content is not valid YAML.
    """
    file_path = Path(file_path).resolve()
    if not file_path.is_file():
        raise FileNotFoundError(f"YAML file '{file_path}' not found.")

    with file_path.open("r", encoding="utf-8") as stream:
        return yaml.safe_load(stream)


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
    script_dir = find_dir_with_file(__file__, "agent_config.json")
    project_dir = find_dir_with_file(script_dir, "env_config.json")
except FileNotFoundError as e:
    raise HTTPException(
        status_code=500,
        detail=f"Required directories for configuration files not found: {e}",
    ) from None

# Load configuration files
try:
    env_config = load_config("env_config.json", project_dir)
    agent_config = load_config("agent_config.json", script_dir)
except HTTPException as e:
    logging.error(f"Error initializing configuration: {e.detail}")
    raise


def streamed_response(openai_response):
    for chunk in openai_response:
        yield chunk.json()


async def async_stream_generator(response) -> AsyncGenerator[str, None]:
    """
    Asynchronously yield the generator output from the response.

    Args:
    response: The response object containing the streaming data.

    Yields:
    str: The chunks of data as they are received.
    """
    async for chunk in response:
        yield chunk.json()
