import json
from typing import AsyncGenerator, Optional
from openai import AsyncOpenAI
from sse_starlette.sse import EventSourceResponse
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

llm_config = dict(host="192.168.2.145",
                  port=16000,
                  route="/v1",
                  model="/llm_models/Qwen2.5-14B-Instruct",
                  )

max_tokens = 8192

url = "http://{host}:{port}{route}".format(host=llm_config.get("host"),
                                           port=llm_config.get("port"),
                                           route=llm_config.get("route")
                                           )


async def async_stream_generator(response) -> AsyncGenerator[str, None]:
    """
    Asynchronously yield the generator output from the response.

    Args:
    response: The response object containing the streaming data.

    Yields:
    str: The chunks of data as they are received.
    """
    async for chunk in response:
        print(f"Chunk received: {chunk.json()}")  # Debugging log
        yield chunk.json()


class Message(BaseModel):
    role: str
    content: str


class ChatCompletion(BaseModel):
    model: str
    messages: list[Message]
    stream: bool = True
    max_tokens: int
    temperature: Optional[float] = 0.1
    top_p: Optional[float] = 0.6


@app.post(path="/v1/chat/completions")
async def generation(request_body: ChatCompletion):
    model = request_body.model
    messages = request_body.messages
    stream = request_body.stream
    temperature = request_body.temperature
    top_p = request_body.top_p
    max_tokens = request_body.max_tokens

    client = AsyncOpenAI(base_url=url,
                         api_key="aaa")

    response = await client.chat.completions.create(model=model,
                                                    messages=messages,
                                                    stream=stream,
                                                    temperature=temperature,
                                                    top_p=top_p,
                                                    max_tokens=max_tokens
                                                    )
    return EventSourceResponse(async_stream_generator(response))
