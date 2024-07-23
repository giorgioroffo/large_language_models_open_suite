<div align="center">
  
[![image](https://github.com/user-attachments/assets/29261802-bb2e-44e4-af8c-c09e843d723b)](https://arxiv.org/html/2407.12036v1)

</div>

<div align="center">

<table>
  <tr>
    <td><a href="https://arxiv.org/html/2407.12036v1" target="_blank">Paper PDF</a></td>
    <td><a href="https://www.researchgate.net/profile/Giorgio-Roffo" target="_blank">Author Page</a></td>
    <td><a href="https://scholar.google.it/citations?user=cD2LFuUAAAAJ&hl=en" target="_blank">Google Scholar</a></td>
    <td><a href="https://github.com/giorgioroffo/large_language_models_open_suite/blob/main/slides_ViT_self_attention_vision_transformer.pdf" target="_blank">Slides Vision Transformer</a></td>
    <td><a href="https://github.com/giorgioroffo/large_language_models_open_suite/blob/main/1_gr_tutorial_LLMs_generative_AI_2024_part1.pdf" target="_blank">Slides LLMs P1</a></td>
    <td><a href="https://github.com/giorgioroffo/large_language_models_open_suite/blob/main/2_gr_tutorial_LLMs_generative_AI_2024_part2.pdf" target="_blank">Slides LLMs P2</a></td>
    <td><a href="https://github.com/giorgioroffo/large_language_models_open_suite/blob/main/3_gr_tutorial_LLMs_generative_AI_2024_part3.pdf" target="_blank">Slides LLMs P3</a></td>
  </tr>
</table>


**Keywords**: Language Model Benchmarking, Pre-Trained LLM Comparison, LLM Performance Analysis, NLP Model Evaluation Tools, Public Dataset Inference for LLMs, BLEU and ROUGE Metrics for LLM, Open Source LLM Testing Tools, Large Language Model Evaluation Software, NLP Benchmarking Suite, Comprehensive LLM Evaluation Toolkit

</div>

## Introduction

The LLMSuite is a public toolbox designed to facilitate practice with large language models (LLMs). It allows users to view code, run inferences, and measure performance using pre-trained models. Currently in version 1, the suite supports evaluation tasks and is under active development to include training functionalities and fine-tuning strategies in future releases.

The toolbox is built with modularity, enabling users to test a variety of models and configurations. Key features include:

- **Model Inference:** Users can select from a range of pre-trained models and run inference tasks to generate and evaluate text outputs.
- **Performance Measurement:** The suite computes and displays performance metrics using established evaluation criteria.
- **Ease of Use:** Designed to be accessible for both researchers and developers, with clear documentation and support for various model architectures.

As the GR LLMSuite evolves, upcoming versions will expand its capabilities to include more advanced functionalities, enhancing its utility for the NLP community.

## Citation

If you use this toolbox in your research or work, please consider citing the following paper:


```bibtex
@misc{roffo2024exploring,
    title={Exploring Advanced Large Language Models with LLMsuite},
    author={Giorgio Roffo},
    year={2024},
    eprint={2407.12036},
    archivePrefix={arXiv},
    primaryClass={cs.CL}
}
```

## Getting Started

To use this toolbox, clone the repository to your local machine and ensure you have the required dependencies installed, including PyTorch and the Hugging Face `transformers` library. Detailed instructions on setting up and running the toolbox, including examples and usage guidelines, can be found in the documentation provided.


To use this toolbox, you need to have cuda installed on your machine. You can download the CUDA Toolkit from the following link:

Official Link: https://developer.nvidia.com/cuda-12-1-0-download-archive

Example installation instructions for Ubuntu 22.04:

```
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-ubuntu2204.pin
sudo mv cuda-ubuntu2204.pin /etc/apt/preferences.d/cuda-repository-pin-600
wget https://developer.download.nvidia.com/compute/cuda/12.1.0/local_installers/cuda-repo-ubuntu2204-12-1-local_12.1.0-530.30.02-1_amd64.deb
sudo dpkg -i cuda-repo-ubuntu2204-12-1-local_12.1.0-530.30.02-1_amd64.deb
sudo cp /var/cuda-repo-ubuntu2204-12-1-local/cuda-*-keyring.gpg /usr/share/keyrings/
sudo apt-get update
sudo apt-get -y install cuda
```

On Ubuntu, you can install the CUDA Toolkit using the following command:

```
sudo apt install nvidia-cuda-toolkit
```
Export the CUDA path to your environment variables using the following commands:

```
export CUDA_HOME=/usr/local/cuda-12
export PATH=$PATH:/usr/local/cuda-12/bin
```
Verify that CUDA has been installed correctly by checking the version of the compiler driver, nvcc, using the following command:

```
bash
nvcc --version
```

The current project is built using PyTorch and Hugging Face's `transformers` library. The code is tested with the following versions: 
```
torch.__version__  = 2.2.1+cu121
torch.version.cuda = 12.1
```

Before installing flash-attn, symlink nvcc to the expected location:
    
``` 
sudo mkdir -p /usr/local/cuda/bin
sudo ln -s /usr/bin/nvcc /usr/local/cuda/bin/nvcc
```

Now, you can install flash-attn using the following command:

```
pip install flash-attn
```


Some extra dependencies are required to run the code. You can install them using the following command:

```
pip install torchmetrics
pip install -q -U einops
pip install -q -U transformers  accelerate peft bitsandbytes
pip install sentencepiece
pip install protobuf

```


## Directory Structure

The toolbox is organized into several key directories and files, each serving a specific purpose in the process of loading data, importing models, conducting evaluations, and utility operations. Below is an overview of the structure and the functionalities of each component:

```
large_language_models_open_suite/
│
├── data_loader/                                    
│   ├── arxiv_summarization_dataset.py
│   └── demo_data_and_tasks.py
│
├── evaluation/                                     
│   └── evaluation_metrics.py
│
├── models/                                         
│   ├── configuration_file.py
│   └── create_models.py
│
├── utils/                                          
│   ├── decoding_strategies.py
│   ├── experiment_settings.py                      
│   ├── output_functions.py                         
│   └── text_preprocessing.py                       
│
├── 1_gr_tutorial_LLMs_generative_AI_2024_part1.pdf 
├── 2_gr_tutorial_LLMs_generative_AI_2024_part2.pdf 
├── 3_gr_tutorial_LLMs_generative_AI_2024_part3.pdf 
├── GRoffo_Introduction_to_LLMs_2024.pdf            
├── LICENSE                                         
├── README.md                                       
└── main_demo.py                                    

```

## Datasets

Another valuable resource is "The PILE," which consists of 20 datasets. These datasets are suitable for small-scale training or fine-tuning models.

<div align="center">
  
[![image](https://github.com/user-attachments/assets/51cdcc57-b6ff-4c7e-9f3b-8430d8b3647f))](https://arxiv.org/html/2407.12036v1)

</div>


Multimodal Large Language Models [My Slides here](https://github.com/giorgioroffo/large_language_models_open_suite/blob/main/multimodal_large_language_models.pdf): A new benchmark is [MME: A Comprehensive Evaluation Benchmark for Multimodal Large Language Models](https://arxiv.org/abs/2306.13394) at the GitHub [MME GitHub link](https://github.com/BradyFU/Awesome-Multimodal-Large-Language-Models/tree/Evaluation).

## RAG (Retrieval-Augmented Generation) based LLM application

LlamaIndex is a framework that facilitates the integration of various data sources and retrieval mechanisms with large language models (LLMs) to create Retrieval-Augmented Generation (RAG) applications. 

```
pip install --upgrade pip
pip install llama_index
pip install torch transformers
```

### General Notions for Readers

Before diving into how to use LlamaIndex in Python, it's important to understand a few key concepts related to vector stores and vector databases.

#### Vector Stores

Vector stores are specialized systems for managing high-dimensional vector embeddings. These embeddings capture the semantic meaning of various data types like text and images. Vector stores are optimized for similarity search, where both the data and queries are represented as vectors, allowing for efficient retrieval of the most relevant information.

#### Vector Databases

Vector databases are designed to store and handle vector embeddings at scale. They provide robust infrastructure for efficient similarity searches across large datasets. Key features include:

- **Efficient Indexing**: Fast retrieval methods for high-dimensional vectors.
- **Similarity Search**: Algorithms to identify vectors most similar to a query.
- **Scalability**: Capability to manage large-scale datasets with billions of vectors.

Examples of vector databases include Pinecone, Milvus, and Faiss.

These concepts form the foundation for understanding how LlamaIndex leverages vector embeddings and retrieval mechanisms to enhance the capabilities of large language models (LLMs) in RAG (Retrieval-Augmented Generation) applications.

## LLMSuite-Demo 1: RAG implementation with LlamaIndex

```
pip install llama-index-vector-stores-faiss
pip install llama-index
pip install llama-index-llms-openai
pip install faiss-cpu
```

The demo Python script **demo_RAG_simple.py** demonstrates how to create, store, and load a vector index using FAISS and the llama_index library. 

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


## LLMSuite-Demo 2: Multi-Modal Retrieval using GPT text embedding and CLIP image embedding for Wikipedia Articles
In *demo_RAG_llama_index.py* we show how to build a Multi-Modal retrieval system using LlamaIndex.
Wikipedia Text embedding index: Generate GPT text embeddings from OpenAI for texts  Wikipedia Images embedding index: CLIP embeddings from OpenAI for images.

```
sudo apt-get update
sudo apt-get install --reinstall ca-certificates
sudo apt-get install python3-tk

pip install torch torchvision
pip install matplotlib scikit-image

pip install llama-index-vector-stores-qdrant
pip install llama_index ftfy regex tqdm
pip install git+https://github.com/openai/CLIP.git
pip install llama-index-embeddings-clip

pip install -U qdrant_client
```

The script will download Wikipedia Images and texts.

```
# Multimodal RAG - Download Wikipedia articles and images
wiki_extracts = util_demo_rag_download_wiki_articles.download_wiki_extracts(folder_name='multimodal_rag')
image_metadata_dict = util_demo_rag_download_wiki_images.download_wiki_images(folder_name='multimodal_rag')
```
![img.png](data/img.png)

Create a script to export you openai private key:

```
pip install --upgrade openai

import private_keys
private_keys

Content: 

import os
os.environ["OPENAI_API_KEY"] = "YOUR_API_KEY"
```    
Make sure your script includes the necessary code to set the API key before invoking any functionality that requires it.

Check if OpenAI is working:

```
import private_keys
private_keys

from openai import OpenAI
client = OpenAI()

completion = client.chat.completions.create(
  model="gpt-4o-mini",
  messages=[
    {"role": "system", "content": "You are a poetic assistant, skilled in explaining complex programming concepts with creative flair."},
    {"role": "user", "content": "Compose a poem that explains the concept of recursion in programming."}
  ]
)

print(completion.choices[0].message)
```

The MLLM processes images using the [https://github.com/OpenAI/CLIP](CLIP transformer) to obtain embeddings.

![image](https://github.com/user-attachments/assets/ad500bf3-dc70-49d9-b028-cdccb6997e31)

*Radford, Alec, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry et al. "Learning transferable visual models from natural language supervision." In International conference on machine learning, pp. 8748-8763. PMLR, 2021*.


## LLMSuite-Demo 3: Fine-Tuning a compare moels 

The demo file '**demo_compare_finetuned_LLMs.py**' shows the differences in performance between two fine-tuned models on the same task. 

## LLMSuite-Demo 4: Fine-Tuning a model on Databricks-Dolly-15k dataset

This script fine-tunes a language model on the Databricks-Dolly-15k dataset using LoRA (Low-Rank Adaptation) technique.

Usage:
    python demo_PEFT_LoRA_finetuning.py -parFile data/param_finetuning_lora.yml

Steps:
1. Parse command-line arguments to get the YAML configuration file.
2. Load configurations including model details, LoRA, and training parameters.
3. Check device availability (GPU/CPU).
4. Create and prepare the dataset for fine-tuning.
5. Configure and load the base model and tokenizer.
6. Set up LoRA and bitsandbytes configurations.
7. Define and initialize training arguments.
8. Train the model using the specified parameters.
9. Save the fine-tuned model.
10. Demonstrate model usage with a sample input.

To ensure the demo **'demo_PEFT_LoRA_finetuning.py'** runs smoothly, please follow these steps to install and upgrade the necessary packages.


For any basic activities like loading models and tokenizers for running inference, upgrading is a must for the newest Gemma model.

```bash
pip install --upgrade datasets
pip install --upgrade transformers
```

For doing efficient operations, install the following packages:

```bash
pip install --upgrade peft
pip install --upgrade trl
pip install bitsandbytes
pip install accelerate
```

For logging and visualizing training progress:

```bash
pip install tensorboard
```

If creating a new dataset, the following package is useful for creating `*.jsonl` files:

```bash
pip install jsonlines
```

Install the `trl` package for training and fine-tuning models:

```bash
pip install trl
```

Example of the training output:

![img.png](data/training.png)

## Conclusions

The GR LLMSuite is a practical framework for benchmarking and evaluating pre-trained language models. It supports comparative analysis across various public datasets using metrics like BLEU and ROUGE. The suite's modular design allows for flexibility in incorporating new models, datasets, and evaluation metrics.
Future enhancements will focus on expanding functionality and usability, guided by community feedback and research advancements.



