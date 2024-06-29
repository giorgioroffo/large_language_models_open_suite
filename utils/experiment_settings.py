# Settings for the experiments can be passed through a YML file or through the command line

# Define percentages of max tokens for each task
max_tokens_by_task = {
    "summarization": 0.05,  # percentage of len based on the prompt
    "translation": 1.0,  # percentage of len based on the prompt
}
