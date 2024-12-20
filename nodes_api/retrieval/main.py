import os
import json
from typing import Dict, Any
from fastapi import FastAPI
from routes import visa_identification, query_expansion, retrieval_milvus_BM25_reranking
import workflow
from utils import load_json_file
import uvicorn

app = FastAPI()

# Load API config
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)

nodes_api_config = load_json_file(os.path.join(parent_dir, "nodes_api_config.json"))

# Validate API config values
host = nodes_api_config["nodes"]["retrieval"]["host"]
port = nodes_api_config["nodes"]["retrieval"]["port"]
api_prefix = nodes_api_config["route_prefix"]

app.include_router(router=workflow.router,
                   prefix=api_prefix
                   )
app.include_router(router=visa_identification.router,
                   prefix=api_prefix
                   )

app.include_router(router=query_expansion.router,
                   prefix=api_prefix
                   )

app.include_router(router=retrieval_milvus_BM25_reranking.router,
                   prefix=api_prefix
                   )


def main():
    uvicorn.run(app="main:app", host=host, port=port, reload=True, reload_includes=["*.json", "*.txt"])


if __name__ == "__main__":
    main()
