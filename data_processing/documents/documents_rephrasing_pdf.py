import os
import asyncio
import aiofiles  # Import aiofiles for async file operations
from motor.motor_asyncio import AsyncIOMotorClient  # Async MongoDB client
from openai import AsyncOpenAI

# MongoDB connection setup
mongodb_uri = "mongodb://gawainng:Aptx4869%60@192.168.2.95:27017/"
mongodb_client = AsyncIOMotorClient(mongodb_uri)

db_name = "Visa_Agent"
coll_names = "Visa_Scheme_Info"
coll = mongodb_client[db_name][coll_names]

# Directory to export rephrased documents
# export_dir = "/mnt/sgnfsdata/gpu_124/gawainng/data/visa/rephrased_data"
export_dir = "/mnt/sgnfsdata/gpu_124/gawainng/data/visa/qwen_rephrased_data/pdf"

# LLM client setup
llm_client = AsyncOpenAI(api_key="aaa", base_url="http://192.168.2.145:30000/v1")
# model = "/llm_models/internlm2_5-20b-chat"
model = "/llm_models/Qwen2.5-32B-Instruct"

task_prompt = """You are an AI assistant specializing in handling visa information for Hong Kong. Your primary task is to rephrase the provided documents, which are presented in a dictionary format, into a clear, coherent, and concise paragraph-long article. Ensure that the rephrased document includes only the information provided, with no additional knowledge or content beyond what is present in the original document. You must rephrase the content using Chinese as the original language.

**Important Requirements:**  
1. Retain and clearly display the document's URL, header, section, and section number within the rephrased content.  
2. Ensure that these structural elements are accurately and logically presented within the paragraph.  
3. Include the source URL of the document and display it at the end of the rephrased content.

{docs}  

**Rephrased Document:**"""


def sanitize_filename(filename: str) -> str:
    """Replace invalid characters in filenames with underscores."""
    return filename.replace("/", "_")


async def process_visa_type(visa_type):
    """Process all headers for a specific visa type."""
    all_headers = await coll.distinct("header", {"visa_type": visa_type})

    for header in all_headers:
        temp_docs = await coll.find({"visa_type": visa_type, "header": header}).to_list(length=None)

        if not temp_docs:
            continue

        task_message = [{"role": "user", "content": task_prompt.format(docs=str(temp_docs))}]

        try:
            response = await llm_client.chat.completions.create(
                model=model,
                messages=task_message,
                stream=False,
                temperature=0.1,
                top_p=0.6
            )

            rephrased_docs = response.choices[0].message.content

            # Sanitize the visa_type and header to avoid invalid characters in file names
            safe_visa_type = sanitize_filename(visa_type)
            safe_header = sanitize_filename(header)

            export_file_name = f"{safe_visa_type}_{safe_header}.txt"
            export_path = os.path.join(export_dir, export_file_name)

            # Write the rephrased content to a file asynchronously
            async with aiofiles.open(export_path, mode='w') as f:
                await f.write(rephrased_docs)

            print(f"Successfully wrote file for Visa Type: {visa_type}, Header: {header}")

        except Exception as e:
            print(f"Error processing Visa Type: {visa_type}, Header: {header}. Error: {e}")


async def main():
    """Main entry point to process all visa types concurrently."""
    all_visa_types = await coll.distinct("visa_type")
    tasks = [process_visa_type(visa_type) for visa_type in all_visa_types]
    await asyncio.gather(*tasks)


if __name__ == "__main__":
    asyncio.run(main())
