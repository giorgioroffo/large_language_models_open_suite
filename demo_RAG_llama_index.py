# RAG (Retrieval-Augmented Generation) based LLM application
'''
Author: Giorgio Roffo, 2024
GitHub: https://github.com/giorgioroffo/large_language_models_open_suite
Report: https://arxiv.org/html/2407.12036v1

If you use this toolbox in your research or work, please consider citing the following paper:

@misc{roffo2024exploring,
    title={Exploring Advanced Large Language Models with LLMsuite},
    author={Giorgio Roffo},
    year={2024},
    eprint={2407.12036},
    archivePrefix={arXiv},
    primaryClass={cs.CL}
}
'''
import os
from pathlib import Path

import certifi
import matplotlib
import torch

from utils import util_demo_rag_download_wiki_articles, util_demo_rag_download_wiki_images
from utils.output_functions import gr_welcome

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
from llama_index.core import StorageContext
from llama_index.core.indices import MultiModalVectorStoreIndex
from llama_index.core.response.notebook_utils import display_source_node
from llama_index.core.schema import ImageNode

device = "cuda:0" if torch.cuda.is_available() else 'cpu'

# Print the welcome message
gr_welcome(device)

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

# QdrantVectorStore: This class represents a vector store that uses Qdrant, an efficient and scalable vector database.
#                    It stores vector embeddings for efficient similarity search.

# VectorStoreIndex: This class provides an interface to interact with a vector store,
#                   allowing for operations such as adding, updating, and querying vectors.

# StorageContext: This class defines the context for storing vectors, managing multiple vector stores (e.g., text and image stores).

# MultiModalVectorStoreIndex: This class extends VectorStoreIndex to handle multiple modalities (e.g., text and image)
#                             using different vector stores.

# Create a local Qdrant client, which will interact with the local Qdrant vector store database.
client = qdrant_client.QdrantClient(path=data_path)

# Initialize a vector store for text embeddings. This store will handle all vector operations for text data.
text_store = QdrantVectorStore(
    client=client, collection_name="text_collection"
)

# Initialize a vector store for image embeddings. This store will handle all vector operations for image data.
image_store = QdrantVectorStore(
    client=client, collection_name="image_collection"
)

# Create a storage context that includes both the text and image vector stores.
# This context will be used to manage the interaction between different modalities (text and image) within the vector store.
storage_context = StorageContext.from_defaults(
    vector_store=text_store, image_store=image_store
)

# Load the documents (text data) from the specified directory.
# SimpleDirectoryReader is used to read text data from the directory.
documents = SimpleDirectoryReader(data_path).load_data()

# Create a MultiModalVectorStoreIndex from the loaded documents and the defined storage context.
# This index will handle both text and image embeddings, allowing for multimodal operations.
index = MultiModalVectorStoreIndex.from_documents(
    documents,
    storage_context=storage_context,
)

# Note: The model that exports the embeddings (e.g., CLIP) is defined within the context of the vector store index operations.
# To use a different model, you would need to modify the embedding generation part of the pipeline.

test_query = "Show me some van Gogh's paintings"
# generate  retrieval results
retriever = index.as_retriever(similarity_top_k=3, image_similarity_top_k=5)
retrieval_results = retriever.retrieve(test_query)

retrieved_image = []
for res_node in retrieval_results:
    if isinstance(res_node.node, ImageNode):
        retrieved_image.append(res_node.node.metadata["file_path"])
    else:
        display_source_node(res_node, source_length=200)

# Plot the retrieved images
util_demo_rag_download_wiki_images.plot_ret_images(retrieved_image)
