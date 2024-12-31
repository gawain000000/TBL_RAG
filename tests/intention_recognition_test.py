import requests

host = "192.168.2.143"
port = 8010
route = "/api/v1/agent/nodes/intention_recognition/intention_recognition"
url = f"http://{host}:{port}{route}"

data = {
    "workflow_state": {},
    "messages": [
        {
            "role": "system",
            "content": "You are an AI assistant developed by Simple AI."
        },
        {
            "role": "user",
            "content": "IANG是什么"
        }
    ],
    "langgraph_path": []
}

try:
    response = requests.post(url=url, json=data)
    response.raise_for_status()  # Raise an HTTPError if the response code is not 200
    res = response.json()  # Parse the response as JSON
    print(res)
except requests.exceptions.RequestException as e:
    print(f"Request failed: {e}")
except ValueError as e:
    print(f"Failed to parse JSON response: {e}")
