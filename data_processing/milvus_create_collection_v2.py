import os
import json
from utils import load_json_file
from pymilvus import MilvusClient, DataType, Collection, utility, db, connections

script_dir = os.path.dirname(os.path.abspath(__file__))
milvus_config = load_json_file(os.path.join(script_dir, "milvus_config.json"))

host = milvus_config["host"]
port = milvus_config["port"]

milvus_uri = f"http://{host}:{port}"
db_name = "visa_agent"
collection_name = "Visa_Scheme_Info"

embedding_dim = 1024
max_document_length = 4096

milvus_connection = connections.connect(host=host, port=port)

client = MilvusClient(uri=milvus_uri)

# client.using_database(db_name=db_name)
# client.drop_collection(collection_name=collection_name)
# quit()

print(client.list_collections())

if db_name not in db.list_database():
    db.create_database(db_name=db_name)

client.using_database(db_name=db_name)
field_config = [
    {"field_name": "id", "datatype": DataType.INT64, "max_length": max_document_length, "is_primary": True,
     "description": "primary"},
    {"field_name": "section_name", "datatype": DataType.VARCHAR, "max_length": max_document_length,
     "description": "section name of the document"},
    {"field_name": "section_number", "datatype": DataType.INT64, "max_length": max_document_length,
     "description": "page number of the document"},
    {"field_name": "visa_type", "datatype": DataType.VARCHAR, "max_length": max_document_length,
     "description": "type of visa"},
    {"field_name": "visa_page_url", "datatype": DataType.VARCHAR, "max_length": max_document_length,
     "description": "the URL link of the document"},
    {"field_name": "content", "datatype": DataType.VARCHAR, "max_length": max_document_length,
     "description": "the chunk of content of the document"},
    {"field_name": "embedding", "datatype": DataType.FLOAT_VECTOR, "dim": embedding_dim,
     "description": "embedding vector"},
]

if not utility.has_collection(collection_name=collection_name):
    # define schema
    schema = MilvusClient.create_schema(auto_id=False, enable_dynamic_field=True)
    for field in field_config:
        schema.add_field(**field)

    # define index
    index_params = client.prepare_index_params()
    index_params.add_index(field_name='embedding',
                           index_type='GPU_CAGRA',
                           metric_type="L2",
                           params={'intermediate_graph_degree': 64,
                                   'graph_degree': 32
                                   }
                           )
    client.create_collection(collection_name=collection_name,
                             schema=schema,
                             index_params=index_params
                             )

print(client.get_load_state(collection_name=collection_name))
print(client.get_collection_stats(collection_name=collection_name))
