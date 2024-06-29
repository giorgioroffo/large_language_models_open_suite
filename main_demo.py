import time

import torch

from data_loader.demo_data_and_tasks import tasks
from evaluation.evaluation_metrics import compute_aggregate_metrics, metric_descriptions, interpret_score
from models.configuration_file import model_configs
from models.create_models import gr_create_model
from utils.decoding_strategies import run_model_task
from utils.experiment_settings import max_tokens_by_task
from utils.output_functions import gr_welcome, print_metrics_table

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

# Initialize containers for generated texts and ground truths
all_generated_texts = {task_name: [] for task_name in tasks.keys()}
all_ground_truths = {task_name: [] for task_name in tasks.keys()}

# Evaluate the model on the prompt
for task_name, task_data in tasks.items():
    print(f"\n{'=' * 60}\n-> Testing the model on task: {task_name}\n{'=' * 60}")
    for data in task_data:
        prompt, gt = data["prompt"], data["gt"]
        print(f'Prompt: {prompt}')
        print(f'Ground Truth: {gt}')

        # Generate text
        start_time = time.time()

        # Dynamic Function Selection and Execution
        generate_llm_output = run_model_task(llm_unique_id, task_name, model_configs)

        generated_text_gr_strategy = generate_llm_output(model, tokenizer, prompt, max_new_tokens=int(
            len(prompt) * max_tokens_by_task[task_name]))

        search_time = time.time() - start_time

        print(f'Generated Text: {generated_text_gr_strategy}')

        # Append generated text and ground truth for later aggregate computation
        all_generated_texts[task_name].append(generated_text_gr_strategy)
        all_ground_truths[task_name].append(gt)

# Print the metrics table for the generated texts and ground truths for each task using the print_metrics_table function
print_metrics_table(tasks, all_generated_texts, all_ground_truths, metric_descriptions, compute_aggregate_metrics,
                    interpret_score)

print('\n\nProcess completed successfully.\n')
