'''
This demo Python script demonstrates how to create, store, and load a vector index using FAISS and the llama_index library. 

### Functionality:
1. Directory Setup: Creates a storage directory in the user's home directory if it doesn't already exist.
2. FAISS Index Initialization: Initializes a FAISS index for storing high-dimensional vectors with a specified dimensionality.
3. Document Loading: Uses the SimpleDirectoryReader to load documents from a specified directory.
4. Vector Store Creation: Creates a FaissVectorStore to manage the FAISS index.
5. Storage Context Initialization: Sets up a StorageContext to manage the vector store and its persistence.
6. Vector Index Creation: Builds a VectorStoreIndex from the loaded documents using the storage context.
7. Index Persistence: Saves the created index to disk in the specified storage directory.
8. Index Loading: Reloads the index from disk, demonstrating how to persist and retrieve the vector index.

### Key Components:
- FAISS: An open-source library that is efficient for similarity search and clustering of dense vectors.
- llama_index: A library providing utilities for managing and querying vector indexes.
- SimpleDirectoryReader: A utility to load documents from a directory.
- VectorStoreIndex: A class representing an index of vectors that supports adding and querying vectors.
- StorageContext: Manages the storage and retrieval of vector data, enabling persistence and loading of vector indexes.
- FaissVectorStore: A specific implementation of a vector store using FAISS for efficient vector operations.

Overall, this script illustrates how to leverage FAISS and llama_index for creating, storing, and querying a vector-based index, which is useful in various applications such as information retrieval, recommendation systems, and more.

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
from pathlib import Path

# RAG and LLAMAINDEX imports
# Create the script 'private_keys.py' in the same directory as this script
# Import the open ai key as follows:
# import os
# os.environ["OPENAI_API_KEY"] = "YOUR_API_KEY"
import private_keys

private_keys

import faiss  # Import the FAISS library, which is used for efficient similarity search and clustering of dense vectors.

from llama_index.core import (
    SimpleDirectoryReader,  # Utility to read documents from a directory.
    load_index_from_storage,  # Function to load an index from storage.
    VectorStoreIndex,  # Class representing an index of vectors.
    StorageContext,  # Class representing the storage context for managing vector stores and indexes.
)
from llama_index.vector_stores.faiss import FaissVectorStore  # FAISS-based implementation of a vector store.
import os  # Import the OS module for interacting with the operating system.
from utils.output_functions import gr_welcome, print_metrics_table
import torch
device = "cuda:0" if torch.cuda.is_available() else 'cpu'

# Print the welcome message
gr_welcome(device)

# Create the folder if it does not exist, using exist_ok=True to avoid errors if it already exists
base_path = 'storage'  # Define the base path where the storage folder will be created.

# Create the full path to the folder in the home directory
home_directory = Path(os.path.expanduser('~'))  # Get the home directory of the current user.
storage_path = home_directory / base_path  # Create the full path to the storage folder.

storage_path.mkdir(exist_ok=True)  # Create the storage directory if it does not exist.

# Dimensions of text-ada-embedding-002
d = 1536  # Define the dimensionality of the text embeddings (1536 dimensions in this case).

faiss_index = faiss.IndexFlatL2(d)  # Create a FAISS index for L2 (Euclidean) distance with the specified dimensions.

# Load documents
documents = SimpleDirectoryReader("./data/").load_data()  # Use SimpleDirectoryReader to load documents from the 'data' directory.

# Create a FAISS-based vector store
vector_store = FaissVectorStore(faiss_index=faiss_index)  # Initialize a FaissVectorStore with the FAISS index.

# Create a storage context with the default settings and the created vector store
storage_context = StorageContext.from_defaults(vector_store=vector_store)  # Create a StorageContext with the default settings.

# Create a vector store index from the loaded documents
index = VectorStoreIndex.from_documents(
    documents, storage_context=storage_context  # Create a VectorStoreIndex using the documents and storage context.
)

# Save the index to disk
index.storage_context.persist(persist_dir=storage_path)  # Persist the storage context to the specified directory.

# Load the index from disk
vector_store = FaissVectorStore.from_persist_dir(storage_path)  # Reload the FaissVectorStore from the persisted directory.
storage_context = StorageContext.from_defaults(
    vector_store=vector_store, persist_dir=storage_path  # Recreate the StorageContext with the reloaded vector store.
)
index = load_index_from_storage(storage_context=storage_context)  # Load the VectorStoreIndex from the storage context.


# Define questions to ask the index
questions = [
    "What are Giorgio Roffo's responsibilities at COSMO IMD?",
    "What was Giorgio Roffo's role in the UCL and Toyota collaboration?",
    "What did Giorgio Roffo work on at the University of Glasgow?",
    "What projects and funding has Giorgio Roffo secured?",
    "Which significant papers has Giorgio Roffo published?"
]

# Query the index and print the responses
for i, question in enumerate(questions):
    query_engine = index.as_query_engine()
    response = query_engine.query(question)

    print(f"Question {i + 1} to the index: {question}")
    print(f"Answer from the index: {response}\n")
