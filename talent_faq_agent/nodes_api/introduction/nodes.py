import os
import requests
from typing import Dict, Any
from talent_faq_agent.nodes_api.utils import load_json_file, nodes_api_config

# Construct file paths
script_dir = os.path.dirname(os.path.abspath(__file__))
introduction_routes_config_path = os.path.join(script_dir, "routes/routes_config.json")

# Load configurations
routes_config = load_json_file(introduction_routes_config_path)

# Extract configuration details
node_name = "introduction"
node_host = nodes_api_config["nodes"][node_name]["host"]
node_port = nodes_api_config["nodes"][node_name]["port"]
route_prefix = nodes_api_config.get("route_prefix", "")
nodes_url = {}

# Create the nodes_url dictionary
nodes_url = {
    route_key: f"http://{node_host}:{node_port}/{route_prefix}{route_info['route']}".replace('//', '/')
    for route_key, route_info in routes_config.items()
}

# Ensure URLs are correctly formatted
nodes_url = {key: url.replace('http:/', 'http://').replace('https:/', 'https://') for key, url in nodes_url.items()}
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


def identification_node(state: Dict) -> Dict[str, Any]:
    return process_node(state, node_name="identification")


def self_introduction_node(state: Dict) -> Dict[str, Any]:
    return process_node(state, node_name="self_introduction")


def simple_ai_introduction_node(state: Dict) -> Dict[str, Any]:
    return process_node(state, node_name="simple_ai_introduction")


def introduction_switch(state: Dict) -> str:
    """
    Determine the next function to execute based on the intention recognition result.

    :param state: The state dictionary containing the intention recognition result.
    :return: The name of the next function to execute.
    :raises RuntimeError: If the intention recognition result is unrecognized.
    """
    state_dict = state["workflow_state"]
    result = state_dict.get("identification_result")
    function_mapping = {
        "AI_chatbot": "self_introduction",
        "SimpleAI": "simple_ai_introduction"
    }

    if result in function_mapping:
        print(function_mapping[result])
        return function_mapping[result]
    else:
        raise RuntimeError(f"Unrecognized intention: '{result}'")
