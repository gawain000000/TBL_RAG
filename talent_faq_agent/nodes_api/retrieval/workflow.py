import os
import logging
from fastapi import APIRouter, Body
from fastapi.responses import JSONResponse
from typing import List, Dict, Any, Optional, Annotated
from pydantic import BaseModel
from talent_faq_agent.nodes_api.utils import GraphState, load_json_file
from langgraph.graph import StateGraph, START, END
from talent_faq_agent.nodes_api.retrieval.nodes import visa_identification_node, query_expansion_node, \
    retrieval_milvus_BM25_reranking_node


def construct_agent():
    # define the graph state
    Talent_FAQ_agent_retrieval_framework = StateGraph(GraphState)

    # add nodes
    Talent_FAQ_agent_retrieval_framework.add_node(node="visa_identification", action=visa_identification_node)
    Talent_FAQ_agent_retrieval_framework.add_node(node="query_expansion", action=query_expansion_node)
    Talent_FAQ_agent_retrieval_framework.add_node(node="retrieval_milvus_BM25_reranking",
                                                  action=retrieval_milvus_BM25_reranking_node)

    # add edges
    Talent_FAQ_agent_retrieval_framework.add_edge(start_key=START, end_key="visa_identification")
    Talent_FAQ_agent_retrieval_framework.add_edge(start_key="visa_identification", end_key="query_expansion")
    Talent_FAQ_agent_retrieval_framework.add_edge(start_key="query_expansion",
                                                  end_key="retrieval_milvus_BM25_reranking")
    Talent_FAQ_agent_retrieval_framework.add_edge(start_key="retrieval_milvus_BM25_reranking", end_key=END)

    # compile the framework
    Talent_FAQ_agent_retrieval = Talent_FAQ_agent_retrieval_framework.compile()
    return Talent_FAQ_agent_retrieval


script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)

router = APIRouter()
logging.basicConfig(level=logging.INFO)

node_name = "retrieval"
nodes_api_config = load_json_file(os.path.join(parent_dir, "nodes_api_config.json"))
retrieval_openapi_examples = load_json_file(os.path.join(script_dir, "openapi_examples/retrieval.json"))

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
async def retrieval(
        request_body: Annotated[NodeState, Body(openapi_examples=retrieval_openapi_examples)]
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
