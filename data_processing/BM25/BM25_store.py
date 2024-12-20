import jieba
from typing import List
from llama_index.retrievers.bm25 import BM25Retriever
from llama_index.core import SimpleDirectoryReader
from llama_index.core.node_parser import SentenceSplitter

chunk_size = 768
chunk_overlap = 100

doc_path = "/mnt/sgnfsdata/gpu_124/gawainng/data/visa/rephrased_data"
documents = SimpleDirectoryReader(input_dir=doc_path, recursive=True).load_data()
BM25_retriever_path = "/mnt/sgnfsdata/gpu_124/gawainng/data/visa/rephrased_visa_bm25"

splitter = SentenceSplitter(chunk_size=chunk_size)
nodes = splitter.get_nodes_from_documents(documents)


def chinese_tokenizer(text: str) -> List[str]:
    # Use jieba to segment Chinese text
    return list(jieba.cut(text))


bm25_retriever = BM25Retriever.from_defaults(
    nodes=nodes,
    similarity_top_k=10,
    tokenizer=chinese_tokenizer
)

bm25_retriever.persist(BM25_retriever_path)
