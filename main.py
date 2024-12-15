from llama_index.core.indices.query.schema import QueryBundle
from llama_index.core import load_index_from_storage, Settings, StorageContext
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader

Settings.embed_model = OllamaEmbedding(model_name="nomic-embed-text")
Settings.llm = Ollama(model="llama3.2", temperature=0.5, request_timeout=120.0)

import os
import time

input_path = input("\n\nEnter relative or absolute path to the document folder >> ")

output_path = f"{input_path}_vectors"

documents = SimpleDirectoryReader(input_path).load_data()

if not os.path.exists(output_path):
    index = VectorStoreIndex.from_documents(documents, show_progress=True)
    index.storage_context.persist(persist_dir=output_path)

storage = StorageContext.from_defaults(persist_dir=output_path)
index = load_index_from_storage(storage)

agent = index.as_query_engine()

while True:
    query = str(input("\nQUERY >> "))
    
    start = time.time()
    res = agent.query(QueryBundle(query_str=query))
    end = time.time()
    
    print("\nRESPONSE >> ", res, "\nResponse time:", round((end-start), 2), " seconds\n\n")