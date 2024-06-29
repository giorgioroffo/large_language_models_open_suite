# LLMSuite: A Python Toolbox for Practicing with Large Language Models

<div align="center">

<table>
  <tr>
    <td><a href="https://github.com/giorgioroffo/large_language_models_open_suite/blob/main/GRoffo_Introduction_to_LLMs_2024.pdf" target="_blank">Paper PDF</a></td>
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
    eprint={5632300},
    archivePrefix={arXiv},
    primaryClass={cs.CL},
    note={A Free Comprehensive Recap of Techniques, Architectures, and Practical Applications},
    url={https://github.com/giorgioroffo/large_language_models_open_suite}
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



## Queries

* "Pre-trained language model comparison toolkit"
* "LLM inference and testing software"
* "NLP model evaluation tools with BLEU ROUGE metrics"
* "Open-source LLM benchmarking suite"
* "Large language model performance analysis toolkit"
* "Code for testing pre-trained language models on public datasets"
* "LLM dataset inference tool"
* "Benchmarking toolkit for NLP models"
* "Software for comparing pre-trained language models"
* "Toolkit for LLM evaluation and metric comparison"


## Conclusions

The GR LLMSuite is a practical framework for benchmarking and evaluating pre-trained language models. It supports comparative analysis across various public datasets using metrics like BLEU and ROUGE. The suite's modular design allows for flexibility in incorporating new models, datasets, and evaluation metrics.
Future enhancements will focus on expanding functionality and usability, guided by community feedback and research advancements.



