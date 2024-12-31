import os
from fastapi import FastAPI
from talent_faq_agent.nodes_api.others_handling.routes import others_handling
from talent_faq_agent.nodes_api.utils import load_json_file
import uvicorn

app = FastAPI()

# Load API config
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)

nodes_api_config = load_json_file(os.path.join(parent_dir, "nodes_api_config.json"))

# Validate API config values
host = nodes_api_config["nodes"]["others_handling"]["host"]
port = nodes_api_config["nodes"]["others_handling"]["port"]
api_prefix = nodes_api_config["route_prefix"]

app.include_router(router=others_handling.router,
                   prefix=api_prefix,
                   tags=["main_node"]
                   )


def main():
    uvicorn.run(app="main:app", host="0.0.0.0", port=port, reload=True, reload_includes=["*.json", "*.txt"])


if __name__ == "__main__":
    main()
