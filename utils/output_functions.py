#!/usr/bin/env python3

import pandas as pd
import private_keys

# Print a cool ASCII art greeting at script start
def gr_welcome(device):
    print("""
            _.-''`.-._
          .'         `.
         |   Large     |
         | Language   |
         |  Model     |
         |   Suite   |
         | v 1.1 2024  |
         |   by       |
         |  Giorgio   | 
         |   Roffo    |
        .'         `.
       /_.-''`.-._/    
    """)

    print("Welcome to the GR Large Language Model Suite! \n")
    print("This toolbox, created by Giorgio Roffo, empowers you to:")
    print(" * Evaluate recent Large Language Models (LLMs) in PyTorch on various tasks.")
    print(" * Run inference on models pre-trained on diverse datasets.")
    print(" * Assess model performance using standard metrics like BLEU and ROUGE.\n")
    print(f"NOTE: The code will be executed on '{device}' device.\n")

    private_keys


def print_metrics_table(tasks, all_generated_texts, all_ground_truths, metric_descriptions, compute_aggregate_metrics,
                        interpret_score):
    """
    Prints a table of metrics scores and their interpretations for each task to the console.

    Args:
    - tasks (dict): A dictionary with task names as keys.
    - all_generated_texts (dict): A dictionary with task names as keys and lists of generated texts as values.
    - all_ground_truths (dict): A dictionary with task names as keys and lists of ground truths as values.
    - metric_descriptions (dict): A dictionary with metric names as keys and their descriptions as values.
    - compute_aggregate_metrics (function): Function that computes aggregate metrics for generated texts and ground truths.
      It should return a dictionary with metric names as keys and scores as values for a given task.
    - interpret_score (function): Function that returns a textual interpretation of a score for a given metric.

    Returns:
    None. This function directly prints the results in a tabular format to the console.
    """
    # Print the descriptions of each metric used
    print("\nMetric Descriptions:")
    for metric, description in metric_descriptions.items():
        print(f"{metric}:\n{description}\n{'-' * 60}\n")

    # Define the column headers for the table
    headers = ["Task Name", "BLEU Score", "ROUGE-L F1 Score", "Score Interpretation"]

    # Determine the width for each column for a neat output
    column_widths = [max(len(header), 20) for header in headers]

    # Print the header row
    header_row = "|".join(header.center(width) for header, width in zip(headers, column_widths))
    print(header_row)
    print("-" * len(header_row))

    # Iterate over each task to compute metrics and print their values
    for task_name in tasks.keys():
        metrics_scores = compute_aggregate_metrics(task_name, all_generated_texts[task_name],
                                                   all_ground_truths[task_name])

        # Initialize a list to hold the string representations of each column's value for the current task
        row_values = [task_name.center(column_widths[0])]

        # Process BLEU and ROUGE-L F1 scores and their interpretations
        for metric_name in ["BLEU", "ROUGE-L F1"]:
            score = metrics_scores.get(metric_name, 0)
            interpretation = interpret_score(metric_name, score)

            # Add the score to the row values, formatting as required
            row_values.append(f"{score:.2f}".center(column_widths[1 if metric_name == "BLEU" else 2]))

        # Add interpretation only once for the last metric processed (assuming similar interpretation for both metrics)
        row_values.append(interpretation.center(column_widths[3]))

        # Print the row for the current task
        print("|".join(row_values))

