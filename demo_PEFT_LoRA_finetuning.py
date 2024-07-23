#!/usr/bin/env python
# coding: utf-8
'''
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

Author: Giorgio Roffo, 2024
GitHub: https://github.com/giorgioroffo/large_language_models_open_suite
Report: https://arxiv.org/html/2407.12036v1

Citation:
@misc{roffo2024exploring,
    title={Exploring Advanced Large Language Models with LLMsuite},
    author={Giorgio Roffo},
    year={2024},
    eprint={2407.12036},
    archivePrefix={arXiv},
    primaryClass={cs.CL}
}
'''
import argparse
import os

import torch
import yaml
from datasets import load_dataset
from peft import LoraConfig, PeftModel
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from trl import SFTTrainer

from utils import util_demo_create_dataset_PEFT_LoRA_finetuning
# Import necessary modules and functions
from utils.output_functions import gr_welcome


def formatting_prompts_func(example):
    """
    Formats prompts from the given example based on specific categories.

    Args:
    example (dict): A dictionary containing lists of instructions, responses, and categories.
                    The expected keys in the dictionary are 'instruction', 'response', and 'category'.

    Returns:
    list: A list of formatted strings. Each string contains an instruction and response
          formatted in a specific way for categories 'open_qa' and 'general_qa'.
    """

    # Initialize an empty list to store the formatted output texts
    output_texts = []

    # Iterate over the length of the instructions in the example
    for i in range(len(example['instruction'])):
        # Check if the category of the current example is either 'open_qa' or 'general_qa'
        if example['category'][i] in ['open_qa', 'general_qa']:
            # Format the text with instruction and response
            text = f"Instruction:\n{example['instruction']}\n\nResponse:\n{example['response']}"
            # Append the formatted text to the output list
            output_texts.append(text)

    # Return the list of formatted output texts
    return output_texts


# Define the main function
def main(config):
    # Determine the device to use (GPU if available, otherwise CPU)
    device = "cuda:0" if torch.cuda.is_available() else 'cpu'

    # Print the welcome message
    gr_welcome(device)

    # Import private keys (if any)
    import private_keys
    private_keys

    # Define some variables - model names
    model_name = config['model_name']
    new_model = config['new_model']

    ################################################################################
    # LoRA parameters
    ################################################################################
    lora_r = config['lora_parameters']['lora_r']
    lora_alpha = config['lora_parameters']['lora_alpha']
    lora_dropout = config['lora_parameters']['lora_dropout']

    ################################################################################
    # bitsandbytes parameters
    ################################################################################
    use_4bit = config['bitsandbytes_parameters']['use_4bit']
    bnb_4bit_compute_dtype = config['bitsandbytes_parameters']['bnb_4bit_compute_dtype']
    bnb_4bit_quant_type = config['bitsandbytes_parameters']['bnb_4bit_quant_type']
    use_nested_quant = config['bitsandbytes_parameters']['use_nested_quant']

    ################################################################################
    # TrainingArguments parameters
    ################################################################################
    output_dir = config['training_arguments']['output_dir']
    num_train_epochs = config['training_arguments']['num_train_epochs']
    fp16 = config['training_arguments']['fp16']
    bf16 = config['training_arguments']['bf16']
    per_device_train_batch_size = config['training_arguments']['per_device_train_batch_size']
    per_device_eval_batch_size = config['training_arguments']['per_device_eval_batch_size']
    gradient_accumulation_steps = config['training_arguments']['gradient_accumulation_steps']
    gradient_checkpointing = config['training_arguments']['gradient_checkpointing']
    max_grad_norm = config['training_arguments']['max_grad_norm']
    learning_rate = config['training_arguments']['learning_rate']
    weight_decay = config['training_arguments']['weight_decay']
    optim = config['training_arguments']['optim']
    lr_scheduler_type = config['training_arguments']['lr_scheduler_type']
    max_steps = config['training_arguments']['max_steps']
    warmup_ratio = config['training_arguments']['warmup_ratio']
    group_by_length = config['training_arguments']['group_by_length']
    save_steps = config['training_arguments']['save_steps']
    logging_steps = config['training_arguments']['logging_steps']

    ################################################################################
    # SFT parameters
    ################################################################################
    max_seq_length = config['sft_parameters']['max_seq_length']
    packing = config['sft_parameters']['packing']
    device_map = config['sft_parameters']['device_map']

    ################################################################################
    # Print message indicating the script's purpose
    print("This script is a demo to fine tune a model on the databricks-dolly-15k dataset using LoRA (PEFT technique)")

    print("Creating the dataset for fine tuning...")
    dataset = util_demo_create_dataset_PEFT_LoRA_finetuning.create_dataset()

    # Load a dataset for fine tuning
    # print("Loading a smaller dataset for demo-fine tuning...")
    # dataset_name = "ai-bites/databricks-mini"
    # dataset = load_dataset(dataset_name, split="train[0:100]")

    # Load QLoRA configuration
    compute_dtype = getattr(torch, bnb_4bit_compute_dtype)

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=use_4bit,  # Activates 4-bit precision loading
        bnb_4bit_quant_type=bnb_4bit_quant_type,  # nf4
        bnb_4bit_compute_dtype=compute_dtype,  # float16
        bnb_4bit_use_double_quant=use_nested_quant,  # False
    )

    # Check GPU compatibility with bfloat16
    if compute_dtype == torch.float16 and use_4bit:
        major, _ = torch.cuda.get_device_capability()
        if major >= 8:
            print("Setting BF16 to True")
            bf16 = True
        else:
            bf16 = False

    # Load base model
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        token=os.environ['HUGGINGFACE_HUB_TOKEN'],
        quantization_config=bnb_config,
        device_map=device_map
    )
    model.config.use_cache = False
    model.config.pretraining_tp = 1

    tokenizer = AutoTokenizer.from_pretrained(model_name,
                                              token=os.environ['HUGGINGFACE_HUB_TOKEN'],
                                              trust_remote_code=True)

    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"  # Fix weird overflow issue with fp16 training

    # Load LoRA configuration
    peft_config = LoraConfig(
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        r=lora_r,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj"]
    )

    # Set training parameters
    training_arguments = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        optim=optim,
        save_steps=save_steps,
        logging_steps=logging_steps,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        fp16=fp16,
        bf16=bf16,
        max_grad_norm=max_grad_norm,
        max_steps=max_steps,
        warmup_ratio=warmup_ratio,
        group_by_length=group_by_length,
        lr_scheduler_type=lr_scheduler_type,
        report_to="tensorboard",
    )

    # Set supervised fine-tuning parameters
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        peft_config=peft_config,
        dataset_text_field="text",
        # formatting_func=format_prompts_fn,
        max_seq_length=max_seq_length,
        tokenizer=tokenizer,
        args=training_arguments,
        packing=packing,
    )

    # Train model
    trainer.train()
    trainer.model.save_pretrained(new_model)

    # Load the trained model
    # fine-tuned model

    input_text = "What should I do on a trip to Europe?"

    base_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        low_cpu_mem_usage=True,
        return_dict=True,
        torch_dtype=torch.float16,
        device_map=device_map,
    )
    model = PeftModel.from_pretrained(base_model, new_model)
    model = model.merge_and_unload()

    # Reload tokenizer to save it
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    input_ids = tokenizer(input_text, return_tensors="pt").to("cuda")
    print(input_ids)
    outputs = model.generate(**input_ids, max_length=128)
    print(tokenizer.decode(outputs[0]))


# Run the main function if this script is executed
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process YAML parameter file.')
    parser.add_argument('-parFile', type=str, required=True, help='Path to the YAML parameter file')
    args = parser.parse_args()

    # Load the YAML file
    with open(args.parFile, 'r') as file:
        config = yaml.safe_load(file)

    main(config)
