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
query_expansion_prompt = load_text_file(os.path.join(parent_dir, "prompts/query_expansion.txt"))
query_expansion_openapi_examples = load_json_file(
    os.path.join(parent_dir, "openapi_examples/query_expansion.json"))

# Extract the route path from configuration
route_path = route_config["query_expansion"]["route"]
llm_generation_config = nodes_api_config.get("llm_generation")


def join_chat_history(chat_history):
    """
    Joins a chat history into a single string with a readable format.

    Args:
        chat_history (list of dict): A list of dictionaries, each containing information about a chat message.
                                     Each dictionary should have keys like 'role' and 'content'.

    Returns:
        str: The joined chat history as a single formatted string.
    """
    joined_history = []
    for message in chat_history:
        role = message.get('role', 'user')  # Default role is 'user'
        if role == 'system':  # Skip system prompts
            continue
        content = message.get('content', '')  # Default content is an empty string
        formatted_message = f"{role.capitalize()}: {content}"
        joined_history.append(formatted_message)

    return "\n".join(joined_history)


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
async def query_expansion(
        request_body: Annotated[NodeState, Body(openapi_examples=query_expansion_openapi_examples)]
):
    # Extract data from request body
    workflow_state = request_body.workflow_state
    historical_messages = request_body.messages
    langgraph_path = request_body.langgraph_path

    all_messages = [{"role": msg.role, "content": msg.content} for msg in historical_messages]
    chat_history = join_chat_history(chat_history=all_messages)
    user_query = all_messages[-1]["content"]

    visa_type = workflow_state["visa_identification_result"]

    task_prompt = query_expansion_prompt.format(chat_history=chat_history,
                                                visa_type=visa_type,
                                                query=user_query
                                                )

    client = AsyncOpenAI(base_url=llm_generation_config["url"],
                         api_key=llm_generation_config["API-key"])

    # print(task_prompt)
    task_messages = [{"role": "user", "content": task_prompt}]

    response = await client.chat.completions.create(model=llm_generation_config["model"],
                                                    messages=task_messages,
                                                    temperature=request_body.temperature,
                                                    top_p=request_body.top_p,
                                                    stream=False
                                                    )

    result = response.choices[0].message.content
    workflow_state["expanded_query"] = result
    langgraph_path.append("query_expansion")
    state = {"workflow_state": workflow_state, "messages": all_messages, "langgraph_path": langgraph_path}
    return JSONResponse(content=state)
