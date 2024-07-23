#!/usr/bin/env python
# coding: utf-8
import os
from pathlib import Path

# Import necessary modules and functions
from data_loader.utils import print_dataset_details
from models.configuration_file import model_configs
from models.create_models import gr_create_model
from datasets import load_dataset
from utils.output_functions import gr_welcome, print_metrics_table
import torch
from tabulate import tabulate  # Ensure that this import is correct
import jsonlines as jl
from collections import defaultdict
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    HfArgumentParser,
    TrainingArguments,
    logging,
)
from peft import LoraConfig, PeftModel
from trl import SFTTrainer


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
def create_dataset():
    # Determine the device to use (GPU if available, otherwise CPU)
    device = "cuda:0" if torch.cuda.is_available() else 'cpu'

    # Print the welcome message
    gr_welcome(device)

    # Import private keys (if any)
    import private_keys
    private_keys


    # Print message indicating the script's purpose
    print("This script is a demo to fine tune a model on the databricks-dolly-15k dataset using LoRA (PEFT technique)")

    # Load a dataset for fine tuning
    print("Loading the databricks-dolly-15k dataset for fine tuning...")
    dataset_name = "databricks/databricks-dolly-15k"

    # Customize the dataset
    dataset = load_dataset(dataset_name, split="train")

    # Print the details of the dataset
    print_dataset_details(dataset)

    categories_count = defaultdict(int)
    for __, data in enumerate(dataset):
        categories_count[data['category']] += 1

    # Step 3: Convert the category counts to a list of lists for tabulate
    table = [[category, count] for category, count in categories_count.items()]

    # Step 4: Print the table using tabulate
    print(tabulate(table, headers=["Category", "Count"], tablefmt="grid"))

    # filter out those that do not have any context
    filtered_dataset = []
    for __, data in enumerate(dataset):
        if data["context"]:
            continue
        else:
            text = f"Instruction:\n{data['instruction']}\n\nResponse:\n{data['response']}"
            filtered_dataset.append({"text": text})

    # Create the full path to the folder in the home directory
    home_directory = Path(os.path.expanduser('~'))  # Get the home directory of the current user.
    datasets_path = home_directory / 'llmsuite_datasets'  # Create the full path to the storage folder.
    datasets_path.mkdir(exist_ok=True)  # Create the storage directory if it does not exist.

    # convert to json and save the filtered dataset as jsonl file
    with jl.open(os.path.join(datasets_path,'dolly-mini-train.jsonl'), 'w') as writer:
        writer.write_all(filtered_dataset[0:])

    print(f'Dataset saved. Path: {datasets_path}/dolly-mini-train.jsonl')

    return filtered_dataset

# Run the main function if this script is executed
if __name__ == "__main__":
    filtered_dataset = create_dataset()
