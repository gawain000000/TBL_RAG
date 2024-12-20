import asyncio
from pymilvus import MilvusClient
from pymongo import MongoClient
from openai import OpenAIError, AsyncOpenAI
from transformers import AutoTokenizer
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Configuration
host = "192.168.2.143"
port = 10006
route_prefix = "/api/v1"
model = "Chuxin-Embedding"
url = f"http://{host}:{port}{route_prefix}"
api_key = "aaa"

# Initialize OpenAI client
openai_client = AsyncOpenAI(base_url=url, api_key=api_key)

# MongoDB setup
mongodb_uri = "mongodb://huiren_chatbot:Hkaift123@192.168.2.95:27017/"
mongodb_client = MongoClient(mongodb_uri)
db_name = "Visa_Agent"
coll_name = "FAQ_zh_CN"
projection = {"_id": 0}

# Milvus setup
milvus_host = "192.168.3.147"
milvus_port = 19530
milvus_uri = f"http://{milvus_host}:{milvus_port}"
milvus_client = MilvusClient(uri=milvus_uri)
milvus_client.using_database(db_name="visa_agent")
milvus_client.get_load_state(collection_name=coll_name)

# Fetch data from MongoDB
all_coll_data = list(mongodb_client[db_name][coll_name].find({}, projection))

# Async function to get embeddings
async def get_embedding(data):
    try:
        query = {
            "model": model,
            "input": data["answer"]
        }
        embedding_response = await openai_client.embeddings.create(**query)
        embedding = embedding_response.data[0].embedding
        return {**data, "embedding": embedding}
    except OpenAIError as e:
        print(f"Error fetching embedding: {e}")
        return None

# Main function
async def main():
    tasks = [get_embedding(query) for query in all_coll_data]
    results = await asyncio.gather(*tasks)
    results = [res for res in results if res is not None]

    # Insert into Milvus
    for i, item in enumerate(results):
        data_to_insert = {**{"id": i}, **item}
        milvus_client.upsert(collection_name=coll_name, data=data_to_insert)
    print("Embeddings inserted into Milvus successfully!")

# Run the async main function
if __name__ == "__main__":
    asyncio.run(main())
