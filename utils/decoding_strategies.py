#!/usr/bin/env python
# Description: This file contains the decoding strategies for text generation using Hugging Face models.

import re

from transformers import PreTrainedModel, PreTrainedTokenizer


def run_model_task(unique_model_id, task, model_configs):
    """
    Retrieves the task-specific function configuration for a given model and task.

    Args:
        unique_model_id (int): The unique identifier or key for the model within the model_configs dictionary.
        task (str): The specific task to retrieve the configuration for (e.g., "text_generation", "summarization").
        model_configs (dict): A dictionary containing the model configurations for different models.

    Returns:
        dict: A dictionary containing the configuration for the specified task and model.

    Raises:
        ValueError: If the unique_model_id does not exist in model_configs or if the specified task is not
        configured for the given model.
    """
    # Check if the unique model ID exists within the model configurations
    if unique_model_id not in model_configs:
        raise ValueError(f"Model ID '{unique_model_id}' not found in model configurations.")

    # Retrieve the model configuration
    model_config = model_configs[unique_model_id]

    # Check if task-specific configurations exist for this model
    if "task_specific_configs" not in model_config or task not in model_config["task_specific_configs"]:
        raise ValueError(f"Task '{task}' not configured for model ID '{unique_model_id}'.")

    # Retrieve and return the task-specific configuration for this model
    return model_config["task_specific_configs"][task]

# Adjust Generation Parameters
# Temperature: Controls randomness. Lower values make the output more deterministic and possibly shorter. Try adjusting this if it's available in your generation function.
# Top-p (nucleus sampling): Sets the cumulative probability threshold for token selection. Lower values can lead to more focused outputs, while higher values encourage diversity.
# Top-k: Limits the number of highest probability tokens considered for each step. A higher value increases diversity.
# No_repeat_ngram_size: Prevents the model from repeating the same n-grams, which can be useful to avoid repetitive outputs.

def handle_flan_t5_small_text_summarizaton(model, tokenizer, input_text, max_new_tokens=50):
    """
    Summarizes a given input text using the FLAN-T5-small model.

    Args:
        model (PreTrainedModel): The pretrained model object from Hugging Face's Transformers library.
            This should be an instance of the FLAN-T5-small model, or any model compatible with the text generation task.
        tokenizer (PreTrainedTokenizer): The tokenizer corresponding to the model. It's used to convert
            the input text into a format suitable for the model (i.e., token IDs).
        input_text (str): The text prompt to generate text from. This is the starting point for the model
            to generate subsequent text.
        max_new_tokens (int): The maximum number of new tokens to generate. Controls the maximum length of the generation.

    Returns:
        List[str]: A list containing the generated text(s) as strings. Each element in the list corresponds
            to a generated sequence based on the input_text. The output is decoded and special tokens are skipped,
            providing clean, readable text.
    """

    # Tokenize the input text and ensure it's on the correct device
    inputs = tokenizer(input_text, return_tensors="pt", padding=True, truncation=True, max_length=16384).to(model.device)

    # Specify generation parameters, including max_new_tokens to control output length and additional parameters
    generation_parameters = {
        "max_new_tokens": max_new_tokens,
        "temperature": 0.9,  # Adjust for more randomness in generation
        "top_p": 0.95,       # Nucleus sampling for diversity
        "no_repeat_ngram_size": 5,  # Avoid repeating the same n-grams
        "pad_token_id": tokenizer.eos_token_id,  # Ensure proper handling of padding during generation
    }

    # Generate outputs based on the provided inputs and generation parameters
    outputs = model.generate(input_ids=inputs['input_ids'], **generation_parameters)

    # Decode the outputs to text, skipping special tokens for readability
    outputs_text = tokenizer.batch_decode(outputs, skip_special_tokens=True)


    if isinstance(outputs_text, list):
        # If the output is a list, return the first element
        return outputs_text[0]
    elif isinstance(outputs_text, str):
        # If the output is already a string, return it directly
        return outputs_text
    else:
        raise ValueError("The output is neither a list of strings nor a single string.")


def handle_t5_v1_1_base_translation(model, tokenizer, input_text, max_new_tokens=50):
    # Encode the input text
    input_ids = tokenizer(input_text, padding=True, truncation=True, max_length=512, return_tensors="pt").input_ids.to(model.device)
    # Generate text output
    generated_ids = model.generate(input_ids, max_length=max_new_tokens)

    # Decode the generated IDs to text
    generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)

    if isinstance(generated_text, list):
        # If the output is a list, return the first element
        return generated_text[0]
    elif isinstance(generated_text, str):
        # If the output is already a string, return it directly
        return generated_text
    else:
        raise ValueError("The output is neither a list of strings nor a single string.")


def handle_t5_v1_1_base_summarization(model, tokenizer, input_text, max_new_tokens=50):
    # Tokenize the text
    input_ids = tokenizer.encode(input_text, padding=True, truncation=True, max_length=512, return_tensors="pt").to(model.device)

    # Generate the summary
    summary_ids = model.generate(input_ids, max_length=max_new_tokens,
                                 min_length=max_new_tokens // 2,
                                 length_penalty=2.0,
                                 num_beams=10,
                                 early_stopping=True)

    # Decode and print the summary
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)

    if isinstance(summary, list):
        # If the output is a list, return the first element
        return summary[0]
    elif isinstance(summary, str):
        # If the output is already a string, return it directly
        return summary
    else:
        raise ValueError("The output is neither a list of strings nor a single string.")


def format_single_prompt_for_mistral(prompt):
    """
    Formats a single prompt string into a conversational structure expected by Mistral.

    Args:
        prompt (str): The single prompt string to be formatted.

    Returns:
        list: A list containing a single message dictionary with the role "user" and the content being the prompt.
    """
    # Assuming the prompt is intended to be a user's message in a conversation
    formatted_input = [{"role": "user", "content": prompt}]

    return formatted_input


def decode_mistral_output(generated_text):
    """
    Decodes the generated text from Mistral, extracting instructions, user inputs, and Mistral responses.

    Args:
        generated_text (str): The generated text string from Mistral model output, containing instructions,
                              user inputs, and responses.

    Returns:
        dict: A dictionary with keys 'instructions', 'user_inputs', and 'responses', each containing a list
              of extracted parts from the generated text.
    """
    decoded_output = {
        "instructions": [],
        "user_inputs": [],
        "responses": []
    }

    # Extract instructions
    decoded_output["instructions"] = re.findall(r"\[INST\] (.*?) \[/INST\]", generated_text)

    # Extract user inputs and responses
    user_inputs_responses = re.findall(r"User: (.*?)\n(.*?)(?=\n|$)", generated_text)
    for user_input, response in user_inputs_responses:
        decoded_output["user_inputs"].append(user_input.strip())
        decoded_output["responses"].append(response.strip())

    return decoded_output

# Warning: The attention mask and the pad token id were not set. As a consequence, you may observe unexpected behavior. Please pass your input's `attention_mask` to obtain reliable results.
def handle_mixtral_8_7B_text_generation(model, tokenizer, input_text, max_new_tokens=50):
    """
    Generates a continuation of the conversation using the Mixtral model.
    - Loading checkpoint shards into the model

    Args:
        model (AutoModelForCausalLM): The pre-trained Mixtral model.
        tokenizer (AutoTokenizer): The tokenizer corresponding to the Mixtral model.
        input_text (list): A list of dictionaries representing the conversation history,
            where each dictionary has a "role" key (user/assistant) and a "content" key.
        max_new_tokens (int): The maximum number of new tokens to generate for the continuation.

    Returns:
        str: The generated continuation of the conversation.
    """
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Format the input text for Mistral
    input_text = format_single_prompt_for_mistral(input_text)

    # Prepare the input for the model
    inputs = tokenizer.apply_chat_template(input_text, padding=True, truncation=True, max_length=512, return_tensors="pt")

    # Ensure inputs are on the same device as the model
    inputs = inputs.to(model.device)

    # Generate a response
    outputs = model.generate(inputs, max_new_tokens=max_new_tokens)

    # Decode the generated tokens to text
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    responses = decode_mistral_output(generated_text)['responses']

    if isinstance(responses, list):
        # If the output is a list, return the first element
        return responses[0]
    elif isinstance(responses, str):
        # If the output is already a string, return it directly
        return responses
    else:
        raise ValueError("The output is neither a list of strings nor a single string.")
