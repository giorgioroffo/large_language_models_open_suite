#!/usr/bin/env python
'''
Arxiv dataset for summarization.
From paper: A Discourse-Aware Attention Model for Abstractive Summarization of Long Documents" by A. Cohan et al.
See: https://aclanthology.org/N18-2097.pdf
See: https://github.com/armancohan/long-summarization

This dataset is designed for the summarization of long documents, specifically adapted from the work presented
in the paper by Cohan et al., 2018. It is adapted for use directly from [this repo](https://github.com/allenai/scicite).
The original data are pre-tokenized, and this version of the dataset processes the text to join tokens with spaces
and adds newline characters for paragraphs to better suit natural text processing tasks.

This dataset is fully compatible with the `run_summarization.py` script from Hugging Face's Transformers library,
provided you include the following line in the `summarization_name_mapping` variable:

```
"ccdv/arxiv-summarization": ("article", "abstract")
```

Data Fields

- `id`: The unique paper ID.
- `article`: A string containing the body of the paper.
- `abstract`: A string containing the abstract of the paper.

Data Splits

The dataset is divided into three splits: train, validation, and test. The token counts are based on white space.

| Dataset Split | Number of Instances | Avg. tokens (article / abstract) |
|---------------|---------------------|----------------------------------|
| Train         | 203,037             | 6038 / 299                       |
| Validation    | 6,436               | 5894 / 172                       |
| Test          | 6,440               | 5905 / 174                       |



Download Link

The dataset can be downloaded from Hugging Face: [ccdv/arxiv-summarization](https://huggingface.co/datasets/ccdv/arxiv-summarization/tree/main)

'''
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset
import numpy as np


class ArxivSummarizationCustomDataset(Dataset):
    def __init__(self, split='train', subset_percentage=None):
        """
        Initialize the dataset.

        Args:
        - split (str): The split of the dataset to load. One of 'train', 'validation', or 'test'.
        - subset_percentage (float, optional): Percentage of the data to load, for quick experiments.
        """
        self.dataset = load_dataset("ccdv/arxiv-summarization", split=split)

        if subset_percentage is not None:
            # Calculate the number of samples to keep
            total_samples = len(self.dataset)
            num_samples = int(total_samples * subset_percentage / 100)

            # Randomly choose the indices of the samples to keep
            indices = np.random.choice(total_samples, num_samples, replace=False)

            # Subset the dataset
            self.dataset = self.dataset.select(indices)

        self.split = split

    def __len__(self):
        """Return the number of samples in the split."""
        return len(self.dataset)

    def __getitem__(self, idx):
        """
        Get a single item from the dataset.

        Args:
        - idx (int): Index of the item.

        Returns:
        - tuple: (article, abstract) where both are strings.
        """
        item = self.dataset[idx]
        article = item['article']
        abstract = item['abstract']
        return article, abstract
