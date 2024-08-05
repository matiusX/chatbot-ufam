import pandas as pd
import os
import dotenv
from chromadb import Client
import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions
from transformers import pipeline

dotenv.load_dotenv()

#loading paths
legis_path = os.getenv("legis_path")
COLLECTION_NAME = os.getenv("collection_name")
files_names = [file_name for file_name in os.listdir(legis_path)]

#create collection
chroma_client = chromadb.PersistentClient(path=COLLECTION_NAME)
sentence_transformer_ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-mpnet-base-v2")
collection = chroma_client.get_or_create_collection(name=COLLECTION_NAME, embedding_function=sentence_transformer_ef)

#indexing documents
documents = []
ids = []
def generate_documents(legis_path):
    global documents, ids
    for file_name in os.listdir(legis_path):
        if file_name.endswith('.txt'):
            file_path = os.path.join(legis_path, file_name)
            try:
                with open(file_path, 'r', encoding='utf-8') as file:
                    content = file.read()
            except UnicodeDecodeError:
                with open(file_path, 'r', encoding='latin1') as file:
                    content = file.read()
            
            documents.append(content)
            ids.append(file_name[:-4])
generate_documents(legis_path)

collection.add(
    documents=documents,
    ids=ids
)

# Retrieval example
'''
question = "Qual é o processo para solicitar uma transferência de curso?"

results = collection.query(
    query_texts=[question],
    n_results=3,
    include=['documents']
)
print(results['documents'])
'''