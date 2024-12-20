import os
from fastapi import HTTPException
from utils import find_dir_with_file, load_json_file

# Load API config
try:
    script_dir = find_dir_with_file(__file__, name="api_config.json")
    project_dir = find_dir_with_file(script_dir, name="env_config.json")
except FileNotFoundError:
    raise HTTPException(status_code=500, detail="Project directory not found.") from None

ROUTE_DIR = os.path.join(script_dir, "routes")

nodes_api_config = load_json_file(os.path.join(script_dir, "nodes_api_config.json"))
env_config = load_json_file(os.path.join(project_dir, "env_config.json"))
route_config = load_json_file(os.path.join(ROUTE_DIR, "route_config.json"))
