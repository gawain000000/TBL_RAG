import os
import logging
from typing import List, Dict, Any, Annotated, Optional
from fastapi import APIRouter, Body
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from talent_faq_agent.nodes_api.utils import load_text_file, load_json_file

router = APIRouter()
logging.basicConfig(level=logging.INFO)

# Load configurations and prompt
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)
route_config = load_json_file(os.path.join(script_dir, "routes_config.json"))
simple_ai_introduction_prompt = load_text_file(os.path.join(parent_dir, "prompts/simple_ai_introduction.txt"))
simple_ai_introduction_openapi_examples = load_json_file(
    os.path.join(parent_dir, "openapi_examples/simple_ai_introduction.json"))

# Extract the route path from configuration
route_path = route_config["simple_ai_introduction"]["route"]


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
async def simple_ai_introduction(
        request_body: Annotated[NodeState, Body(openapi_examples=simple_ai_introduction_openapi_examples)]
):
    # Extract data from request body
    workflow_state = request_body.workflow_state
    historical_messages = request_body.messages
    langgraph_path = request_body.langgraph_path

    all_messages = [{"role": msg.role, "content": msg.content} for msg in historical_messages]
    user_query = all_messages[-1]["content"]

    workflow_state["generation_prompt"] = simple_ai_introduction_prompt.format(user_query=user_query)
    langgraph_path.append("ai_introduction_generation")
    state = {"workflow_state": workflow_state, "messages": all_messages, "langgraph_path": langgraph_path}
    return JSONResponse(content=state)
