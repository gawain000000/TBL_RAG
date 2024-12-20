from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai_like import OpenAILike
from llama_index.core import Settings, SimpleDirectoryReader, StorageContext, VectorStoreIndex
from llama_index.vector_stores.milvus import MilvusVectorStore

# llm_config = dict(base_url="http://192.168.2.143:20000/v1",
#                   api_key="aaa",
#                   model="/llm_models/internlm2_5-7b-chat"
#                   )

llm_config = dict(base_url="http://192.168.2.143:20000/v1",
                  api_key="aaa",
                  model="/llm_models/Qwen2.5-14B-Instruct"
                  )

embedding_config = dict(base_url="http://192.168.2.143:30000/v1",
                        api_key="aaa",
                        model="/embedding_models/Chuxin-Embedding"
                        )

host = "192.168.2.143"
port = 19530
milvus_uri = f"http://{host}:{port}"

llm = OpenAILike(api_key=llm_config["api_key"],
                 api_base=llm_config["base_url"],
                 model=llm_config["model"],
                 max_tokens=1024,
                 context_window=3900,
                 is_chat_model=True,
                 is_function_calling_model=True,
                 temperature=0.1,
                 top_p=0.6
                 )

embed_model = OpenAIEmbedding(api_key=embedding_config["api_key"],
                              api_base=embedding_config["base_url"],
                              model_name=embedding_config["model"])

chunk_size = 800
chunk_overlap = 100

Settings.llm = llm
Settings.embed_model = embed_model
Settings.chunk_size = chunk_size
Settings.chunk_overlap = chunk_overlap

doc_path = "/mnt/sgnfsdata/gpu_124/gawainng/data/visa/rephrased_data"
documents = SimpleDirectoryReader(input_dir=doc_path, recursive=True).load_data()

nodes = Settings.node_parser.get_nodes_from_documents(documents)

vector_store = MilvusVectorStore(uri=milvus_uri,
                                 collection_name="visa_rag_agent_v4",
                                 dim=1024,
                                 overwrite=True,
                                 enable_sparse=False,
                                 embedding_field="doc_embedding",
                                 similarity_metric="L2",
                                 index_config={"index_type": "GPU_CAGRA",
                                               "intermediate_graph_degree": 64,
                                               "graph_degree": 32
                                               },
                                 # hybrid_ranker="RRFRanker",
                                 # hybrid_ranker_params={"k": 60},
                                 )

storage_context = StorageContext.from_defaults(vector_store=vector_store)
storage_context.docstore.add_documents(nodes)
vector_index = VectorStoreIndex.from_documents(documents=documents,
                                               storage_context=storage_context,
                                               )
