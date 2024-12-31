from langgraph.graph import StateGraph, START, END
from talent_faq_agent.agent_api.utils import GraphState
from talent_faq_agent.agent_api.nodes import (
    intention_recognition_node, intention_switch,
    retrieval_node, introduction_node, others_handling_node
)

# define the graph state
Talent_FAQ_agent_framework = StateGraph(GraphState)

# Add nodes
## intention recognition
Talent_FAQ_agent_framework.add_node(node="intention_recognition", action=intention_recognition_node)

## retrieval
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
Talent_FAQ_agent_framework.add_edge(start_key="retrieval", end_key=END)

## self introduction
Talent_FAQ_agent_framework.add_edge(start_key="introduction", end_key=END)

## others handling
Talent_FAQ_agent_framework.add_edge(start_key="others_handling", end_key=END)

# compile the framework
Talent_FAQ_agent = Talent_FAQ_agent_framework.compile()
