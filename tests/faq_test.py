from openai import OpenAI
from llama_index.retrievers.bm25 import BM25Retriever
from llama_index.core.retrievers import QueryFusionRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.response_synthesizers import Refine
from llama_index.llms.openai_like import OpenAILike

# llm_config = dict(host="192.168.2.143",
#                   port=10020,
#                   route_prefix="/api/v1",
#                   route="/agent/talent_faq_chatbot",
#                   model="/llm_models/internlm2_5-7b-chat"
#                   )

llm_config = dict(host="192.168.2.143",
                  port=20000,
                  route_prefix="/api/v1",
                  route="/v1",
                  model="/llm_models/Qwen2.5-14B-Instruct"
                  )

# host = "192.168.2.145"
# port = 23333
# route = "/v1"


url = "http://{host}:{port}{route}".format(host=llm_config.get("host"),
                                           port=llm_config.get("port"),
                                           route=llm_config.get("route")
                                           )

# model = "InternVL2-40B"

# prompt = "什么是高才通"
# prompt = "什么是高端人才通行证计划"
# prompt = "内地居民如果想在香港工作，需要满足哪些条件？"
# prompt = "write a 1000 words article about AI"
prompt = "who are u?"
# prompt = "who develop u?"
# prompt = "who is Simple AI?"

messages = [
    {"role": "system", "content": "You are an AI assistant developed by Simple AI."},
    {"role": "user", "content": prompt}
]

client = OpenAI(base_url=url,
                api_key="aaa"
                )

stream = True

response = client.chat.completions.create(model=llm_config["model"],
                                          messages=messages,
                                          stream=stream
                                          )

if stream:
    for chunk in response:
        print(chunk.choices[0].delta.content or "", end="", flush=True)
else:
    print(f"{response.choices[0].message.content}")
