import os
import sys
import json
import logging
from typing import List, Dict, Any, Annotated, Optional
from fastapi import APIRouter, Body
from fastapi.responses import JSONResponse
from pydantic import BaseModel

# Load configurations and prompt
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)
grandparent_dir = os.path.dirname(parent_dir)

sys.path.append(parent_dir)

from utils import load_text_file, load_json_file

router = APIRouter()
logging.basicConfig(level=logging.INFO)

route_config = load_json_file(os.path.join(script_dir, "routes_config.json"))
nodes_api_config = load_json_file(os.path.join(grandparent_dir, "nodes_api_config.json"))
self_introduction_prompt = load_text_file(os.path.join(parent_dir, "prompts/self_introduction.txt"))
self_introduction_openapi_examples = load_json_file(os.path.join(parent_dir, "openapi_examples/self_introduction.json"))

# Extract the route path from configuration
route_path = route_config["self_introduction"]["route"]


# Define Pydantic models
class Message(BaseModel):
    """Pydantic model to represent a single message."""
    role: str
    content: str


class NodeState(BaseModel):
    """Pydantic model for the request body, representing the conversation state."""
    workflow_state: Dict[str, Any]
    messages: List[Message]
    langgraph_path: List[str]
    temperature: Optional[float] = 0.1
    top_p: Optional[float] = 0.6


@router.post(path=route_path)
async def self_introduction(
        request_body: Annotated[NodeState, Body(openapi_examples=self_introduction_openapi_examples)]
):
    # Extract data from request body
    workflow_state = request_body.workflow_state
    historical_messages = request_body.messages
    langgraph_path = request_body.langgraph_path

    all_messages = [{"role": msg.role, "content": msg.content} for msg in historical_messages]
    user_query = all_messages[-1]["content"]

    workflow_state["generation_prompt"] = self_introduction_prompt.format(user_query=user_query)
    langgraph_path.append("retrieval_generation")
    state = {"workflow_state": workflow_state, "messages": all_messages, "langgraph_path": langgraph_path}
    return JSONResponse(content=state)
