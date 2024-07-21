'''
Contrastive Language–Image Pretraining

CLIP (Contrastive Language–Image Pretraining) is a model developed by OpenAI that combines image and text processing
capabilities. It uses a Transformer-based architecture to create embeddings for both text and images,
allowing it to perform various tasks such as image classification, zero-shot learning, and retrieval tasks without
needing task-specific training.

CLIP consists of two main components:
- An image encoder, which processes images and generates embeddings.
- A text encoder, which processes text and generates embeddings.

Both the image and text encoders are trained jointly using a contrastive learning objective.
This means that the model learns to associate images with their corresponding text descriptions and differentiate
them from unrelated text descriptions.

'''
import private_keys
import os
from pathlib import Path
import torch
from PIL import Image
from torchvision import transforms
import openai
import clip

from utils import util_demo_rag_download_wiki_articles, util_demo_rag_download_wiki_images

# Import the API key
private_keys

# Define the path to the image directory
data_wiki_path = 'multimodal_rag'
home_directory = Path(os.path.expanduser('~'))
data_path = home_directory / data_wiki_path

# Multimodal RAG - Download Wikipedia articles and images
wiki_extracts = util_demo_rag_download_wiki_articles.download_wiki_extracts(data_path=data_path)
image_metadata_dict = util_demo_rag_download_wiki_images.download_wiki_images(data_path=data_path)

print('Downloaded Wikipedia articles and images.')

# Load the CLIP model
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

# Process images and obtain embeddings
image_files = list(data_path.glob("*.jpg"))
for image_file in image_files:
    image = Image.open(image_file)
    image_tensor = preprocess(image).unsqueeze(0).to(device)

    with torch.no_grad():
        image_features = model.encode_image(image_tensor)

    # Save the embeddings
    embedding_file = data_path / f"{image_file.stem}_embedding.pt"
    torch.save(image_features.cpu(), embedding_file)
    print(f"Saved embedding for {image_file.name} to {embedding_file}")

print("All embeddings saved successfully.")
