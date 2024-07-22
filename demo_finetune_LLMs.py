#!/usr/bin/env python
# coding: utf-8

# Import necessary libraries
import itertools
import jsonlines
from datasets import load_dataset
from pprint import pprint

import private_keys
private_keys

from lamini.runners.basic_model_runner import BasicModelRunner
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSeq2SeqLM

# Define prompt templates
prompt_template_without_input = """Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Response:"""

# Function to load and process the dataset
def load_and_process_dataset(dataset_name, m):
    # Load the dataset
    dataset = load_dataset(dataset_name, split="train", streaming=True)

    # Select top m examples
    top_m = list(itertools.islice(dataset, m))
    processed_data = []

    # Process each example with appropriate prompt template
    for j in top_m:
        processed_prompt = prompt_template_without_input.format(instruction=j["instruction"])

        processed_data.append({"input": processed_prompt, "output": j["output"]})

    return processed_data

# Function to save processed data to jsonl file
def save_to_jsonl(data, filename):
    with jsonlines.open(filename, 'w') as writer:
        writer.write_all(data)

# Function to run model inference
def inference(text, model, tokenizer, max_input_tokens=1000, max_output_tokens=100):
    # Tokenize the input text
    input_ids = tokenizer.encode(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=max_input_tokens
    )

    # Generate output tokens
    device = model.device
    generated_tokens_with_prompt = model.generate(
        input_ids=input_ids.to(device),
        max_length=max_output_tokens
    )

    # Decode the generated text
    generated_text_with_prompt = tokenizer.batch_decode(generated_tokens_with_prompt, skip_special_tokens=True)

    # Strip the prompt from the generated text
    generated_text_answer = generated_text_with_prompt[0][len(text):]

    return generated_text_answer

# Function to compare non-instruction-tuned vs. instruction-tuned models
def compare_models():
    # Load non-instruction-tuned model
    non_instruct_model = BasicModelRunner("meta-llama/Llama-2-7b-hf")
    non_instruct_output = non_instruct_model("Tell me how to train my dog to sit")
    print("Not instruction-tuned output (Llama 2 Base):", non_instruct_output)

    # Load instruction-tuned model
    instruct_model = BasicModelRunner("meta-llama/Llama-2-7b-chat-hf")
    instruct_output = instruct_model("Tell me how to train my dog to sit")
    print("Instruction-tuned output (Llama 2):", instruct_output)

# Main function
def main():
    import private_keys
    private_keys

    # Load and process the instruction-tuned dataset
    dataset_name = "tatsu-lab/alpaca"
    m = 5
    processed_data = load_and_process_dataset(dataset_name, m)

    # Print processed data
    pprint(processed_data[0])

    # Save processed data to jsonl file
    save_to_jsonl(processed_data, 'alpaca_processed.jsonl')

    # Compare non-instruction-tuned vs. instruction-tuned models
    compare_models()

    # Load tokenizer and smaller model
    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/pythia-70m")
    model = AutoModelForCausalLM.from_pretrained("EleutherAI/pythia-70m")

    # Load fine-tuning dataset
    finetuning_dataset_path = "lamini/lamini_docs"
    finetuning_dataset = load_dataset(finetuning_dataset_path)
    print(finetuning_dataset)

    # Test sample inference
    test_sample = finetuning_dataset["test"][0]
    print(test_sample)

    print('> Baseline model, test sample inference:')
    print(inference(test_sample["question"], model, tokenizer))

    # Load fine-tuned model and perform inference
    instruction_model = AutoModelForCausalLM.from_pretrained("lamini/lamini_docs_finetuned")
    print('> Fine-tuned model, test sample inference:')
    print(inference(test_sample["question"], instruction_model, tokenizer))


# Execute main function
if __name__ == "__main__":
    main()
