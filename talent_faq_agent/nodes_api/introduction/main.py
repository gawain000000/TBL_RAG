import os
from fastapi import FastAPI
from talent_faq_agent.nodes_api.introduction.routes import self_introduction, identification, simple_ai_introduction
from talent_faq_agent.nodes_api.introduction import workflow
from talent_faq_agent.nodes_api.utils import load_json_file
import uvicorn

app = FastAPI()

# Load API config
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)

nodes_api_config = load_json_file(os.path.join(parent_dir, "nodes_api_config.json"))

# Validate API config values
host = nodes_api_config["nodes"]["introduction"]["host"]
port = nodes_api_config["nodes"]["introduction"]["port"]
api_prefix = nodes_api_config["route_prefix"]

app.include_router(router=workflow.router,
                   prefix=api_prefix,
                   tags=["main_node"]
                   )
app.include_router(router=identification.router,
                   prefix=api_prefix,
                   tags=["sub_nodes"]
                   )
app.include_router(router=self_introduction.router,
                   prefix=api_prefix,
                   tags=["sub_nodes"]
                   )
app.include_router(router=simple_ai_introduction.router,
                   prefix=api_prefix,
                   tags=["sub_nodes"]
                   )


def main():
    uvicorn.run(app="main:app", host="0.0.0.0", port=port, reload=True, reload_includes=["*.json", "*.txt"])


if __name__ == "__main__":
    main()
