# RAG (Retrieval-Augmented Generation) based LLM application
import os
from pathlib import Path

import certifi
import torch


import utils
from models.configuration_file import model_configs
from models.create_models import gr_create_model
from utils import util_demo_rag_download_wiki_articles, util_demo_rag_download_wiki_images
from utils.output_functions import gr_welcome
import matplotlib
matplotlib.use('TkAgg')

import httpx
client = httpx.Client(verify=False)

# RAG and LLAMAINDEX imports
# Create the script 'private_keys.py' in the same directory as this script
# Import the open ai key as follows:
# import os
# os.environ["OPENAI_API_KEY"] = "YOUR_API_KEY"
import private_keys
private_keys


import qdrant_client
from llama_index.core import SimpleDirectoryReader
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.core import VectorStoreIndex, StorageContext
from llama_index.core.indices import MultiModalVectorStoreIndex


device = "cuda:0" if torch.cuda.is_available() else 'cpu'

# Print the welcome message
gr_welcome(device)


# Extract all available models from the configurations
available_models = []
for config_data in model_configs.values():
    available_models.extend(config_data["model_names"])

print("+ Available models:")
for i, models in enumerate(available_models):
    print(f"{i + 1}. {models}")

# Prompt the user to select a model by its number
try:
    print('\n')
    user_input = int(input("Enter the number of the model you'd like to test: ")) - 1  # Adjust for 0-based index
    if user_input >= 0 and user_input < len(available_models):
        selected_model_name = available_models[user_input]
    else:
        raise ValueError("Selected model number is out of range.")
except ValueError as e:
    # Handle the case where the input is not an integer or out of range
    selected_model_name = 'gpt2'

# Initialize a variable to store the description of the selected model
selected_model_description = None

# Search for the model configuration that matches the selected model name
for config in model_configs.values():
    if selected_model_name in config["model_names"]:
        selected_model_description = config["description"]
        break

# Check if the description was found
if selected_model_description:
    print(f"\n+ Selected model: {selected_model_name}")
    print(f"+ Description: {selected_model_description}")
else:
    print("Model description not found.")

# Create the language model using gr_create_model function
model, tokenizer, llm_unique_id = gr_create_model(model_name=selected_model_name)

# Set the model to evaluation mode
model.eval()

data_wiki_path = 'multimodal_rag'
# Create the full path to the folder in the home directory
home_directory = Path(os.path.expanduser('~'))
data_path = home_directory / data_wiki_path

# Multimodal RAG - Download Wikipedia articles and images
wiki_extracts = util_demo_rag_download_wiki_articles.download_wiki_extracts(data_path=data_path)
image_metadata_dict = util_demo_rag_download_wiki_images.download_wiki_images(data_path=data_path)

print('Downloaded Wikipedia articles and images.')
util_demo_rag_download_wiki_images.plot_images(image_metadata_dict)

print('Build Multi Modal Vector Store using Text and Image embeddings under different collections.')

# Configure HTTP client to use certifi's CA bundle
httpx_client = httpx.Client(verify=certifi.where())


# Create a local Qdrant vector store
client = qdrant_client.QdrantClient(path="qdrant_db")

text_store = QdrantVectorStore(
    client=client, collection_name="text_collection"
)
image_store = QdrantVectorStore(
    client=client, collection_name="image_collection"
)
storage_context = StorageContext.from_defaults(
    vector_store=text_store, image_store=image_store
)

# Create the MultiModal index
documents = SimpleDirectoryReader(data_path).load_data()
index = MultiModalVectorStoreIndex.from_documents(
    documents,
    storage_context=storage_context,
)
