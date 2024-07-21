# Example Tutorial
# Multi-Modal Retrieval using GPT text embedding and CLIP image embedding for Wikipedia Articles
import os
from pathlib import Path

import requests


def download_wiki_extracts(data_path):

    # Create the folder if it does not exist, using exist_ok=True to avoid errors if it already exists
    data_path.mkdir(exist_ok=True)

    wiki_titles = [
        "batman",
        "Vincent van Gogh",
        "San Francisco",
        "iPhone",
        "Tesla Model S",
        "BTS",
    ]

    # Dictionary to store Wikipedia extracts
    wiki_extracts = {}

    for title in wiki_titles:
        response = requests.get(
            "https://en.wikipedia.org/w/api.php",
            params={
                "action": "query",
                "format": "json",
                "titles": title,
                "prop": "extracts",
                "explaintext": True,
            },
        ).json()
        page = next(iter(response["query"]["pages"].values()))
        wiki_text = page["extract"]

        # Save the extract to the dictionary
        wiki_extracts[title] = wiki_text

        with open(data_path / f"{title}.txt", "w") as fp:
            fp.write(wiki_text)

    return wiki_extracts