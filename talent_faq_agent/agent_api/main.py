import os
from fastapi import FastAPI, Body
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Optional, Annotated, AsyncGenerator
from openai import OpenAI, AsyncOpenAI
from sse_starlette.sse import EventSourceResponse
from talent_faq_agent.agent_api.utils import load_json_file, async_stream_generator, load_text_file, env_config
from talent_faq_agent.agent_api.agent_architecture import Talent_FAQ_agent
import logging
import uvicorn

# Setup logging
logging.basicConfig(level=logging.INFO)

script_dir = os.path.dirname(os.path.abspath(__file__))

agent_config_path = os.path.join(script_dir, "agent_config.json")
agent_config = load_json_file(agent_config_path)
llm_config = env_config.get("llm_generation")

Talent_FAQ_agent_examples = load_json_file(os.path.join(script_dir, "openapi_examples/Talent_FAQ_agent.json"))
system_prompt = load_text_file(os.path.join(script_dir, "prompts/system.txt"))
generation_prompt = load_text_file(os.path.join(script_dir, "prompts/generation.txt"))


def join_text_with_numbering(text_list):
    """
    Joins the text in the list with numbering starting from 1.
    Each section of text will be prefixed with a number followed by a period.

    Args:
    text_list (list of str): List of text to be joined.

    Returns:
    str: The formatted and joined text with numbering.
    """
    formatted_text = ""
    for i, text in enumerate(text_list, start=1):
        formatted_text += f"{i}. {text.strip()}\n\n"
    return formatted_text.strip()


host = agent_config["host"]
port = agent_config["port"]
route_prefix = agent_config["route_prefix"]
route_path = agent_config["route"]

app = FastAPI()


class Message(BaseModel):
    role: str
    content: str


class AgentChatCompletion(BaseModel):
    model: str
    messages: list[Message]
    stream: bool = True
    temperature: Optional[float] = 0.1
    top_p: Optional[float] = 0.6


@app.post(path=route_path)
async def Agent_generation(
        request_body: Annotated[AgentChatCompletion, Body(openapi_examples=Talent_FAQ_agent_examples)]
):
    try:
        model = request_body.model
        messages = request_body.messages
        stream = request_body.stream
        temperature = request_body.temperature
        top_p = request_body.top_p

        # Treat all other messages as `historical_message`
        historical_messages = [{"role": msg.role, "content": msg.content} for msg in messages]

        # Ensure the first message is always the system prompt
        system_message = {"role": "system", "content": system_prompt}
        if not historical_messages or historical_messages[0].get("role") != "system":
            historical_messages.insert(0, system_message)
        elif historical_messages[0]["content"] != system_prompt:
            historical_messages[0] = system_message

        graph_state = {"workflow_state": {},
                       "messages": historical_messages,
                       "langgraph_path": []
                       }

        logging.info(f"Graph State: {graph_state}")
        agent_response = Talent_FAQ_agent.invoke(graph_state)
        logging.info(f"Agent Response: {agent_response}")

        workflow_state = agent_response.get("workflow_state")
        if workflow_state["intention_recognition_result"] == "information_asking":
            expanded_user_query = workflow_state["expanded_query"]
            retrieved_docs = workflow_state["documents_retrieval_result"]
            retrieved_info = join_text_with_numbering(retrieved_docs)

            task_messages = generation_prompt.format(user_query=expanded_user_query, chunks=retrieved_info)
        else:
            task_messages = workflow_state["generation_prompt"]

        generation_messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": task_messages}
        ]

        client = AsyncOpenAI(base_url=llm_config["url"],
                             api_key=llm_config["API-key"])

        response = await client.chat.completions.create(model=model,
                                                        messages=generation_messages,
                                                        stream=stream,
                                                        temperature=temperature,
                                                        top_p=top_p
                                                        )

        # Handle streamed responses
        if stream:
            return EventSourceResponse(async_stream_generator(response))

        # Handle non-streamed response
        return response
    except Exception as e:
        logging.error(f"Error during chat completion: {e}")
        return JSONResponse(content={"error": str(e)}, status_code=500)


def main():
    uvicorn.run(app=app, host="0.0.0.0", port=port)


if __name__ == '__main__':
    main()
