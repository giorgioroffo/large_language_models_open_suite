''' FineTuning RAG with CrossEncoder for QASPER Dataset

Dataset: Download QASPER Dataset from HuggingFace Hub.

Sample Extraction:

- Extract 800 samples from training data.
- Extract 80 samples from test data.

Training Data: Format 800 training samples for CrossEncoder fine-tuning with questions, contexts, and relevance scores (0 or 1).

Evaluation Data:

- Rag Eval Dataset: Includes research paper content, questions, and long answers.
- Reranking Eval Dataset: Includes research paper content, questions, and relevant contexts.
- Model Fine-Tuning: Fine-tune CrossEncoder and push the model to HuggingFace Hub.

Evaluation:

- Test using OpenAI embeddings with/without rerankers.
- Use Hits Metric and Pairwise Comparison Evaluator for evaluation.
'''
from datasets import load_dataset
import random


# Download QASPER dataset from HuggingFace https://huggingface.co/datasets/allenai/qasper
dataset = load_dataset("allenai/qasper")

# Split the dataset into train, validation, and test splits
train_dataset = dataset["train"]
validation_dataset = dataset["validation"]
test_dataset = dataset["test"]

random.seed(42)  # Set a random seed for reproducibility

# Randomly sample 800 rows from the training split
train_sampled_indices = random.sample(range(len(train_dataset)), 800)
train_samples = [train_dataset[i] for i in train_sampled_indices]


# Randomly sample 100 rows from the test split
test_sampled_indices = random.sample(range(len(test_dataset)), 80)
test_samples = [test_dataset[i] for i in test_sampled_indices]

# Now we have 800 research papers for training and 80 research papers to evaluate on

