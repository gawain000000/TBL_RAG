import os
import sys
import logging
from fastapi import APIRouter, Body
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Dict, Any, Annotated, Optional
from llama_index.core import VectorStoreIndex, Settings, ChatPromptTemplate
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.vector_stores.milvus import MilvusVectorStore
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai_like import OpenAILike
from llama_index.retrievers.bm25 import BM25Retriever
from llama_index.core.retrievers import QueryFusionRetriever
from llama_index.core.postprocessor.llm_rerank import LLMRerank
import nest_asyncio

# Load configurations and prompt
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)
grandparent_dir = os.path.dirname(parent_dir)

sys.path.append(parent_dir)

from utils import load_text_file, load_json_file

router = APIRouter()
logging.basicConfig(level=logging.INFO)

route_config = load_json_file(os.path.join(script_dir, "routes_config.json"))
nodes_api_config = load_json_file(os.path.join(grandparent_dir, "nodes_api_config.json"))
retrieval_milvus_BM25_openapi_examples = load_json_file(
    os.path.join(parent_dir, "openapi_examples/retrieval_milvus_BM25_reranking.json"))

route_path = route_config["retrieval_milvus_BM25_reranking"]["route"]
llm_config = nodes_api_config.get("llm_generation")
embedding_config = nodes_api_config.get("embedding")
embedding_api_url = "http://{host}:{port}{route}".format(host=embedding_config["host"],
                                                         port=embedding_config["port"],
                                                         route=embedding_config["route"]
                                                         )

milvus_config = nodes_api_config.get("milvus")
llamaindex_config = nodes_api_config.get("llamaindex")
BM25_persist_path = llamaindex_config["BM_25_persist"]

# llamaindex config
llm = OpenAILike(api_key=llm_config["API-key"],
                 api_base=llm_config["url"],
                 model=llm_config["model"],
                 # max_tokens=1024,
                 context_window=3900,
                 is_chat_model=True,
                 is_function_calling_model=True,
                 )

embedding_url = "http://{host}:{port}{route}".format(host=embedding_config["host"],
                                                     port=embedding_config["port"],
                                                     route=embedding_config["route"]
                                                     )
embed_model = OpenAIEmbedding(api_key=embedding_config["API-key"],
                              api_base=embedding_url,
                              model_name=embedding_config["model"])

Settings.llm = llm
Settings.embed_model = embed_model
Settings.chunk_size = llamaindex_config["chunk_size"]
Settings.chunk_overlap = llamaindex_config["chunk_overlap"]

vector_store = MilvusVectorStore(uri=milvus_config["uri"],
                                 collection_name=milvus_config["visa_rag_agent"],
                                 dim=milvus_config["config"]["dim"],
                                 overwrite=False,
                                 enable_sparse=False,
                                 embedding_field=milvus_config["config"]["embedding_field"],
                                 similarity_metric=milvus_config["config"]["similarity_metric"],
                                 index_config=milvus_config["config"]["index_config"],
                                 # hybrid_ranker="RRFRanker",
                                 # hybrid_ranker_params={"k": 60},
                                 )
milvus_index = VectorStoreIndex.from_vector_store(vector_store=vector_store)
milvus_retriever = milvus_index.as_retriever(similarity_top_k=milvus_config["similarity_top_k"])

bm25_retriever = BM25Retriever.from_persist_dir(BM25_persist_path)
nest_asyncio.apply()
retriever = QueryFusionRetriever(retrievers=[milvus_retriever,
                                             bm25_retriever
                                             ],
                                 retriever_weights=[0.6, 0.4],
                                 similarity_top_k=milvus_config["similarity_top_k"],
                                 num_queries=4,
                                 use_async=True,
                                 mode="dist_based_score"
                                 )
ranker = LLMRerank(choice_batch_size=10, top_n=3, llm=llm)


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


@router.post(path=route_path)
async def retrieval_milvus_BM25(
        request_body: Annotated[NodeState, Body(openapi_examples=retrieval_milvus_BM25_openapi_examples)]
):
    # Extract data from request body
    workflow_state = request_body.workflow_state
    historical_messages = request_body.messages
    langgraph_path = request_body.langgraph_path

    all_messages = [{"role": msg.role, "content": msg.content} for msg in historical_messages]

    # user_query = all_messages[-1]["content"]
    expanded_user_query = workflow_state["expanded_query"]

    retrieved_results = retriever.retrieve(expanded_user_query)
    reranked_retrieval_results = ranker.postprocess_nodes(retrieved_results, query_str=expanded_user_query)
    retrieved_text = []
    for item in reranked_retrieval_results:
        print("=" * 100)
        print(item.get_content())
        retrieved_text.append(item.get_content())

    workflow_state["documents_retrieval_result"] = retrieved_text
    langgraph_path.append("documents_retrieval")
    state = {"workflow_state": workflow_state, "messages": all_messages, "langgraph_path": langgraph_path}
    return JSONResponse(content=state)