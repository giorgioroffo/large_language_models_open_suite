import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, T5ForConditionalGeneration, AutoModelForCausalLM, \
    T5Tokenizer

from utils import decoding_strategies

# Model configurations for the GR Large Language Model Suite
model_configs = {
    0: {
        "description": "Configuration for FLAN T5, FLAN-T5 was released in the paper Scaling Instruction-Finetuned Language Models.",
        "model_names": ["google/flan-t5-small"],
        "model_class": AutoModelForSeq2SeqLM,  # Specify the class for causal language models
        "tokenizer_class": AutoTokenizer,
        "config": None,  # Use the GPT2Config if specific configuration needed, or None for default
        "additional_configs": {
            "device_map": "auto",
            "torch_dtype": torch.bfloat16,
        },
        "task_specific_configs": {
            'summarization': decoding_strategies.handle_flan_t5_small_text_summarizaton,
            'translation': decoding_strategies.handle_flan_t5_small_text_summarizaton,
        },
    },
    1: {
        "description": "Configuration for T5-Small, a versatile Seq2Seq model for tasks like summarization, translation, and more.",
        "model_names": ["t5-small"],
        "model_class": T5ForConditionalGeneration,  # Specify the class for Seq2Seq models
        "tokenizer_class": T5Tokenizer,
        "config": None,  # Use the T5Config if specific configuration needed, or None for default
        "additional_configs": {
            "device_map": "auto",
            "torch_dtype": torch.bfloat16,
        },
        "task_specific_configs": {
            'summarization': decoding_strategies.handle_t5_v1_1_base_summarization,
            'translation': decoding_strategies.handle_t5_v1_1_base_translation,
        },
    },
    2: {
        "description": "Configuration for Mixtral-8x7B-Instruct-v0.1 - chat_template.",
        "model_names": ["mistralai/Mixtral-8x7B-Instruct-v0.1"],
        "model_class": AutoModelForCausalLM,
        "tokenizer_class": AutoTokenizer,
        "config": None,
        "additional_configs": {
            "device_map": "auto",
            "torch_dtype": torch.bfloat16,
        },
        "task_specific_configs": {
            'summarization': decoding_strategies.handle_mixtral_8_7B_text_generation,
            'translation': decoding_strategies.handle_mixtral_8_7B_text_generation,
        },
    },
}
