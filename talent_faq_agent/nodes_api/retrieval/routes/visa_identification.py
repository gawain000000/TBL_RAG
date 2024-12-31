import os
import json
import logging
from pydantic import BaseModel
from openai import AsyncOpenAI
from typing import List, Dict, Any, Optional, Annotated
from fastapi import APIRouter, HTTPException, Body
from fastapi.responses import JSONResponse
from talent_faq_agent.nodes_api.utils import load_json_file, load_text_file, env_config

# Initialize the API router and logging
router = APIRouter()
logging.basicConfig(level=logging.INFO)

# Load configurations and prompt
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)
route_config = load_json_file(os.path.join(script_dir, "routes_config.json"))
visa_identification_prompt = load_text_file(os.path.join(parent_dir, "prompts/visa_identification.txt"))
visa_identification_openapi_examples = load_json_file(
    os.path.join(parent_dir, "openapi_examples/visa_identification.json"))

# Extract the route path from configuration
route_path = route_config["visa_identification"]["route"]
llm_generation_config = env_config.get("llm_generation")


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


class VisaIdentificationType(BaseModel):
    visa_type: str


@router.post(path=route_path)
async def intention_recognition(
        request_body: Annotated[NodeState, Body(openapi_examples=visa_identification_openapi_examples)]
):
    try:
        # Extract data from request body
        workflow_state = request_body.workflow_state
        historical_messages = request_body.messages
        langgraph_path = request_body.langgraph_path

        # Prepare chat history for the AI model
        chat_history = "\n".join([f"{msg.role}: {msg.content}" for msg in historical_messages if msg.role != "system"])
        task_messages = [
            {"role": "system", "content": visa_identification_prompt},
            {"role": "user", "content": chat_history}
        ]

        # API call to OpenAI's chat completion
        client = AsyncOpenAI(base_url=llm_generation_config["url"],
                             api_key=llm_generation_config["API-key"])

        # response = await client.beta.chat.completions.parse(
        #     model=nodes_api_config["llm_generation"]["model"],
        #     messages=task_messages,
        #     temperature=request_body.temperature,
        #     top_p=request_body.top_p,
        #     response_format=VisaIdentificationType
        # )
        #
        # print(response.choices[0].message.parsed)
        # result = {"visa_type": response.choices[0].message.parsed.visa_type}

        response = await client.chat.completions.create(
            model=llm_generation_config["model"],
            messages=task_messages,
            temperature=request_body.temperature,
            top_p=request_body.top_p
        )
        print(response.choices[0].message.content)
        result = json.loads(response.choices[0].message.content)

        # Update state with AI response
        workflow_state["visa_identification_result"] = result.get("visa_type", "unknown")
        langgraph_path.append("visa_identification")

        # Return updated state
        state = {
            "workflow_state": workflow_state,
            "messages": [{"role": msg.role, "content": msg.content} for msg in historical_messages],
            "langgraph_path": langgraph_path
        }
        return JSONResponse(content=state)

    except Exception as e:
        logging.error(f"Error in AI completion: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error in AI completion: {str(e)}")
