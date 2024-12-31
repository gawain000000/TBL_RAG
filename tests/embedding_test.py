from openai import OpenAI
from llama_index.vector_stores.milvus import MilvusVectorStore
from llama_index.core import StorageContext


# llm_config = dict(host="192.168.2.143",
#                   port=10006,
#                   route="/api/v1",
#                   model="Chuxin-Embedding"
#                   )

llm_config = dict(host="192.168.2.143",
                  port=30000,
                  route="/v1",
                  model="/embedding_models/Chuxin-Embedding"
                  )

url = "http://{host}:{port}{route}".format(host=llm_config.get("host"),
                                           port=llm_config.get("port"),
                                           route=llm_config.get("route")
                                           )

client = OpenAI(base_url=url,
                api_key="aaa"
                )

embedding_input = "内地居民如果想在香港工作，需要满足哪些条件？"

response = client.embeddings.create(model=llm_config["model"],
                                    input=embedding_input
                                    )
print(response.data[0].embedding)
