from langgraph.graph import StateGraph, START, END
from utils import GraphState
from agent_nodes import (
    intention_recognition_node, intention_switch,
    # query_expansion_node, document_retrieval_node, document_retrieval_llamaindex_node,
    # document_retrieval_milvus_BM25_node, documents_grading_node, retrieval_generation_node,
    retrieval_node,
    introduction_node, others_handling_node
)

# define the graph state
Talent_FAQ_agent_framework = StateGraph(GraphState)

# Add nodes
## intention recognition
Talent_FAQ_agent_framework.add_node(node="intention_recognition", action=intention_recognition_node)

## retrieval
# Talent_FAQ_agent_framework.add_node(node="documents_retrieval", action=document_retrieval_node)
# Talent_FAQ_agent_framework.add_node(node="documents_retrieval_llamaindex", action=document_retrieval_llamaindex_node)
# Talent_FAQ_agent_framework.add_node(node="query_expansion", action=query_expansion_node)
# Talent_FAQ_agent_framework.add_node(node="documents_retrieval_milvus_BM25", action=document_retrieval_milvus_BM25_node)
# Talent_FAQ_agent_framework.add_node(node="documents_grading", action=documents_grading_node)
# Talent_FAQ_agent_framework.add_node(node="retrieval_generation", action=retrieval_generation_node)
Talent_FAQ_agent_framework.add_node(node="retrieval", action=retrieval_node)

## introduction
Talent_FAQ_agent_framework.add_node(node="introduction", action=introduction_node)

## others handling
Talent_FAQ_agent_framework.add_node(node="others_handling", action=others_handling_node)

# Add edges for product_information_query flow
## starting
Talent_FAQ_agent_framework.add_edge(start_key=START, end_key="intention_recognition")
Talent_FAQ_agent_framework.add_conditional_edges(source="intention_recognition", path=intention_switch)

## retrieval
# Talent_FAQ_agent_framework.add_edge(start_key="documents_retrieval", end_key="documents_grading")
# Talent_FAQ_agent_framework.add_edge(start_key="documents_retrieval_llamaindex", end_key="documents_grading")
# Talent_FAQ_agent_framework.add_edge(start_key="query_expansion", end_key="documents_retrieval_milvus_BM25")
# Talent_FAQ_agent_framework.add_edge(start_key="documents_retrieval_milvus_BM25", end_key="documents_grading")
# Talent_FAQ_agent_framework.add_edge(start_key="documents_grading", end_key="retrieval_generation")
# Talent_FAQ_agent_framework.add_edge(start_key="retrieval_generation", end_key=END)
Talent_FAQ_agent_framework.add_edge(start_key="retrieval", end_key=END)

## self introduction
Talent_FAQ_agent_framework.add_edge(start_key="introduction", end_key=END)

## others handling
Talent_FAQ_agent_framework.add_edge(start_key="others_handling", end_key=END)

# compile the framework
Talent_FAQ_agent = Talent_FAQ_agent_framework.compile()
