import json
import logging
from typing import Dict, Any
from pathlib import Path


# Helper functions
def load_json_file(file_path: str) -> Dict[str, Any]:
    """Load and return content from a JSON file."""
    try:
        with open(file_path, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        logging.error(f"{file_path} not found.")
        raise RuntimeError(f"{file_path} file not found.")
    except json.JSONDecodeError:
        logging.error(f"Failed to decode JSON from {file_path}")
        raise RuntimeError(f"Error decoding JSON from {file_path}")


def load_text_file(file_path: str) -> str:
    """Load and return text content from a file."""
    try:
        with open(file_path, "r") as f:
            return f.read()
    except FileNotFoundError:
        logging.error(f"{file_path} not found.")
        raise RuntimeError(f"{file_path} file not found.")


def find_dir_with_file(
        path: str | Path,
        name: str,
) -> str:
    """
    Find a directory and parents of it containing a specific file.
    """
    path = Path(path)
    if (path / name).exists():
        return str(path)

    for parent in path.parents:
        if (parent / name).exists():
            return str(parent)

    raise FileNotFoundError
