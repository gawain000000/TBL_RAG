import os
import json
import requests
from typing import Dict, Any
from utils import load_json_file, find_dir_with_file

# Construct file paths
script_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = find_dir_with_file(path=script_dir, name="env_config.json")
nodes_api_dir = os.path.join(project_dir, "nodes_api")

# Configuration file paths
nodes_api_config_path = os.path.join(nodes_api_dir, "nodes_api_config.json")
## simple nodes config
intention_recognition_routes_config_path = os.path.join(nodes_api_dir, "intention_recognition",
                                                        "routes/routes_config.json")
# retrieval_routes_config_path = os.path.join(nodes_api_dir, "retrieval", "routes/routes_config.json")
others_handling_routes_config_path = os.path.join(nodes_api_dir, "others_handling", "routes/routes_config.json")

# Load configurations
nodes_api_config = load_json_file(nodes_api_config_path)
## simple nodes
intention_recognition_routes_config = load_json_file(intention_recognition_routes_config_path)
# retrieval_routes_config = load_json_file(retrieval_routes_config_path)
others_handling_config = load_json_file(others_handling_routes_config_path)
## subgraph nodes
introduction_routes_config = {"introduction": nodes_api_config.get("nodes").get("introduction")}
retrieval_routes_config = {"retrieval": nodes_api_config.get("nodes").get("retrieval")}


# Combine node route configurations
nodes_routes = {
    "intention_recognition": intention_recognition_routes_config,
    "retrieval": retrieval_routes_config,
    "introduction": introduction_routes_config,
    "others_handling": others_handling_config
}

# Extract configuration details
nodes_url_config = nodes_api_config.get("nodes", {})
route_prefix = nodes_api_config.get("route_prefix", "")
nodes_url = {}


def construct_node_urls():
    """
    Constructs the complete URLs for all nodes based on the configuration.
    """
    for group, nodes in nodes_routes.items():
        # Get host and port for the group
        group_config = nodes_url_config.get(group, {})
        host = group_config.get("host", "localhost")
        port = group_config.get("port", 80)

        for node_name, node_details in nodes.items():
            # Construct the URL
            route = node_details.get("route", "/")
            url = f"http://{host}:{port}{route_prefix}{route}"
            nodes_url[node_name] = url


# Construct all node URLs at initialization
construct_node_urls()

print(nodes_url)


def request_node(state: Dict, node_name: str) -> Dict[str, Any]:
    """
    Sends a POST request to the specified node route with the given state.

    :param state: The state dictionary to send as JSON.
    :param node_name: The node name for the request.
    :return: JSON response from the node.
    :raises RuntimeError: If the request fails.
    """
    url = nodes_url.get(node_name)
    if not url:
        raise ValueError(f"Node URL for '{node_name}' not found in configuration.")

    try:
        with requests.post(url=url, json=state, stream=True) as response:
            response.raise_for_status()  # Raise an HTTPError for bad responses
            return response.json()
    except requests.RequestException as e:
        raise RuntimeError(f"Request to node '{node_name}' failed: {e}")


def process_node(state: Dict, node_name: str) -> Dict[str, Any]:
    """
    Generic function to process requests for different nodes.

    :param state: The state dictionary to send.
    :param node_name: The name of the node to request.
    :return: JSON response from the node.
    """
    return request_node(state, node_name)


# Specific node functions
def intention_recognition_node(state: Dict) -> Dict[str, Any]:
    return process_node(state, node_name="intention_recognition")


# def query_expansion_node(state: Dict) -> Dict[str, Any]:
#     return process_node(state, node_name="query_expansion")
#
#
# def document_retrieval_node(state: Dict) -> Dict[str, Any]:
#     return process_node(state, node_name="documents_retrieval")
#
#
# def document_retrieval_llamaindex_node(state: Dict) -> Dict[str, Any]:
#     return process_node(state, node_name="documents_retrieval_llamaindex")
#
#
# def document_retrieval_milvus_BM25_node(state: Dict) -> Dict[str, Any]:
#     return process_node(state, node_name="documents_retrieval_milvus_BM25")
#
#
# def documents_grading_node(state: Dict) -> Dict[str, Any]:
#     return process_node(state, node_name="documents_grading")
#
#
# def retrieval_generation_node(state: Dict) -> Dict[str, Any]:
#     return process_node(state, node_name="retrieval_generation")

def retrieval_node(state: Dict) -> Dict[str, Any]:
    return process_node(state, node_name="retrieval")


def introduction_node(state: Dict) -> Dict[str, Any]:
    return process_node(state, node_name="introduction")


def others_handling_node(state: Dict) -> Dict[str, Any]:
    return process_node(state, node_name="others_handling")


def intention_switch(state: Dict) -> str:
    """
    Determine the next function to execute based on the intention recognition result.

    :param state: The state dictionary containing the intention recognition result.
    :return: The name of the next function to execute.
    :raises RuntimeError: If the intention recognition result is unrecognized.
    """
    state_dict = state["workflow_state"]
    result = state_dict.get("intention_recognition_result")
    function_mapping = {
        # "information_asking": "documents_retrieval_llamaindex",
        "information_asking": "retrieval",
        "introduction": "introduction",
        "not_relevant": "others_handling"
    }

    if result in function_mapping:
        print("*" * 100)
        print(function_mapping[result])
        return function_mapping[result]
    else:
        raise RuntimeError(f"Unrecognized intention: '{result}'")
