import os
import logging
from fastapi import APIRouter, Body
from fastapi.responses import JSONResponse
from typing import List, Dict, Any, Optional, Annotated
from pydantic import BaseModel
from talent_faq_agent.nodes_api.utils import GraphState, load_json_file
from langgraph.graph import StateGraph, START, END
from talent_faq_agent.nodes_api.introduction.nodes import identification_node, self_introduction_node, simple_ai_introduction_node, introduction_switch


def construct_agent():
    # define the graph state
    Talent_FAQ_agent_introduction_framework = StateGraph(GraphState)

    # add nodes
    Talent_FAQ_agent_introduction_framework.add_node(node="identification", action=identification_node)
    Talent_FAQ_agent_introduction_framework.add_node(node="self_introduction", action=self_introduction_node)
    Talent_FAQ_agent_introduction_framework.add_node(node="simple_ai_introduction", action=simple_ai_introduction_node)

    # add edges
    Talent_FAQ_agent_introduction_framework.add_edge(start_key=START, end_key="identification")
    Talent_FAQ_agent_introduction_framework.add_conditional_edges(source="identification", path=introduction_switch)
    Talent_FAQ_agent_introduction_framework.add_edge(start_key="self_introduction", end_key=END)
    Talent_FAQ_agent_introduction_framework.add_edge(start_key="simple_ai_introduction", end_key=END)

    # compile the framework
    Talent_FAQ_agent_introduction = Talent_FAQ_agent_introduction_framework.compile()
    return Talent_FAQ_agent_introduction


script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)

router = APIRouter()
logging.basicConfig(level=logging.INFO)

node_name = "introduction"
nodes_api_config = load_json_file(os.path.join(parent_dir, "nodes_api_config.json"))
introduction_openapi_examples = load_json_file(os.path.join(script_dir, "openapi_examples/introduction.json"))

node_config = nodes_api_config.get("nodes", {}).get(node_name, {})
host = node_config["host"]
port = node_config["port"]
route_path = node_config["route"]

agent = construct_agent()


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
async def introduction(
        request_body: Annotated[NodeState, Body(openapi_examples=introduction_openapi_examples)]
):
    workflow_state = request_body.workflow_state
    messages = request_body.messages
    langgraph_path = request_body.langgraph_path

    all_messages = [{"role": msg.role, "content": msg.content} for msg in messages]

    graph_state = {
        "workflow_state": workflow_state,
        "messages": all_messages,
        "langgraph_path": langgraph_path,
        "temperature": request_body.temperature,
        "top_p": request_body.top_p
    }
    agent_response = await agent.ainvoke(graph_state)
    return JSONResponse(content=agent_response)
