import os
import sys
import json
import logging
from typing import List, Dict, Any, Annotated, Optional
from fastapi import APIRouter, Body
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from openai import AsyncOpenAI

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
identification_prompt = load_text_file(os.path.join(parent_dir, "prompts/identification.txt"))
identification_openapi_examples = load_json_file(os.path.join(parent_dir, "openapi_examples/identification.json"))

# Extract the route path from configuration
route_path = route_config["identification"]["route"]


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


class IdentificationClasses(BaseModel):
    identification: str


@router.post(path=route_path)
async def identification(
        request_body: Annotated[NodeState, Body(openapi_examples=identification_openapi_examples)]
):
    # Extract data from request body
    workflow_state = request_body.workflow_state
    historical_messages = request_body.messages
    langgraph_path = request_body.langgraph_path

    # Convert historical messages to chat history format
    all_messages = [{"role": msg.role, "content": msg.content} for msg in historical_messages]

    # Filter out any existing "system" messages from all_messages to avoid duplication
    filtered_messages = [msg for msg in all_messages if msg["role"] != "system"]

    # Prepare chat history for the AI model
    task_messages = [
        {"role": "system", "content": identification_prompt},
        *filtered_messages
    ]

    client = AsyncOpenAI(base_url=nodes_api_config["llm_generation"]["url"],
                         api_key=nodes_api_config["llm_generation"]["API-key"])

    # response = await client.beta.chat.completions.parse(
    #     model=nodes_api_config["llm_generation"]["model"],
    #     messages=task_messages,
    #     temperature=request_body.temperature,
    #     top_p=request_body.top_p,
    #     response_format=IdentificationClasses
    # )
    # result = {"identification": response.choices[0].message.parsed.identification}

    response = await client.chat.completions.create(
        model=nodes_api_config["llm_generation"]["model"],
        messages=task_messages,
        temperature=request_body.temperature,
        top_p=request_body.top_p
    )
    print(response.choices[0].message.content)

    result = json.loads(response.choices[0].message.content)

    # Update state with AI response
    workflow_state["identification_result"] = result.get("identification", "unknown")
    langgraph_path.append("identification")

    # Return updated state
    state = {
        "workflow_state": workflow_state,
        "messages": all_messages,
        "langgraph_path": langgraph_path
    }
    return JSONResponse(content=state)
