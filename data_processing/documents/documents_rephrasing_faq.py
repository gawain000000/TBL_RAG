import os
import asyncio
import aiofiles  # Import aiofiles for async file operations
from motor.motor_asyncio import AsyncIOMotorClient  # Async MongoDB client
from openai import AsyncOpenAI

# MongoDB connection setup
mongodb_uri = "mongodb://gawainng:Aptx4869%60@192.168.2.95:27017/"
mongodb_client = AsyncIOMotorClient(mongodb_uri)

db_name = "Visa_Agent"
coll_names = "FAQ_zh_CN"
coll = mongodb_client[db_name][coll_names]

# Directory to export rephrased documents
# export_dir = "/mnt/sgnfsdata/gpu_124/gawainng/data/visa/rephrased_data/FAQ"
export_dir = "/mnt/sgnfsdata/gpu_124/gawainng/data/visa/qwen_rephrased_data/FAQ"


# LLM client setup
llm_client = AsyncOpenAI(api_key="aaa", base_url="http://192.168.2.145:30000/v1")
# model = "/llm_models/internlm2_5-20b-chat"
model = "/llm_models/Qwen2.5-32B-Instruct"

task_prompt = """You are an AI assistant specializing in processing visa information for Hong Kong. Your task is to transform an FAQ, provided in a dictionary format, into a clear, coherent paragraph written in Chinese. The rephrased document must strictly adhere to the given information, ensuring accuracy and completeness without introducing any additional details or context.

**Key Requirements:**  
1. **Category, Question, and Answer**: Clearly highlight the document category, question, and answer as core components of the paragraph.  
2. **Logical Structure**: Present the information in a logical, well-organized manner to ensure smooth readability and natural flow.  
3. **Source Reference**: Include the source URL at the end of the document.  

**Original Document:**  
{docs}  

**Rephrased Document (in Chinese):**"""


def sanitize_filename(filename: str) -> str:
    """Replace invalid characters in filenames with underscores."""
    return filename.replace("/", "_")


async def process_document(data):
    """Processes a single document from MongoDB, rephrases it, and writes it to a file."""
    category_1 = data["category_1"]
    category_2 = data["category_2"]
    task_message = [{"role": "user", "content": task_prompt.format(docs=str(data))}]

    response = await llm_client.chat.completions.create(
        model=model,
        messages=task_message,
        stream=False,
        temperature=0.1,
        top_p=0.6
    )

    rephrased_document = response.choices[0].message.content
    file_name = f"{sanitize_filename(category_1)}_{sanitize_filename(category_2)}.txt"
    file_path = os.path.join(export_dir, file_name)

    async with aiofiles.open(file_path, "w") as f:
        await f.write(rephrased_document)


async def main():
    """Main function to iterate through all documents in MongoDB and process them asynchronously."""
    projection = {"_id": 0}
    async for data in coll.find({}, projection):
        await process_document(data)


if __name__ == "__main__":
    asyncio.run(main())