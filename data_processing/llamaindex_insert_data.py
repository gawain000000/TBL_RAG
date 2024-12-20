from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai_like import OpenAILike
from llama_index.core.node_parser import SemanticSplitterNodeParser, SentenceSplitter
from llama_index.core import SimpleDirectoryReader, SimpleDirectoryReader, ChatPromptTemplate
from llama_index.core import VectorStoreIndex, StorageContext, Settings
from llama_index.vector_stores.milvus import MilvusVectorStore
from llama_index.core.retrievers import VectorIndexRetriever

doc_path = "/mnt/sgnfsdata/gpu_145/vivianye/jupyter_code/Visa_Agent/Visa_Documents/Files"
documents = SimpleDirectoryReader(doc_path).load_data()
