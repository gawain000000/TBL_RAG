import os
import json
import yaml
from pathlib import Path
import itertools
from typing import Dict, Any, Union


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
