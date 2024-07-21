# Example Tutorial
# Multi-Modal Retrieval using GPT text embedding and CLIP image embedding for Wikipedia Articles
import os
import os
import os
import urllib.request
import urllib.request
from pathlib import Path
from pathlib import Path

import matplotlib.pyplot as plt
import wikipedia
import wikipedia
from PIL import Image


def plot_images(image_metadata_dict):
    original_images_urls = []
    images_shown = 0
    for image_id in image_metadata_dict:
        img_path = image_metadata_dict[image_id]["img_path"]
        if os.path.isfile(img_path):
            filename = image_metadata_dict[image_id]["filename"]
            image = Image.open(img_path).convert("RGB")

            plt.subplot(8, 8, len(original_images_urls) + 1)
            plt.imshow(image)
            plt.xticks([])
            plt.yticks([])

            original_images_urls.append(filename)
            images_shown += 1
            if images_shown >= 64:
                break

    plt.tight_layout()
    plt.show()


def plot_ret_images(image_paths):
    images_shown = 0
    plt.figure(figsize=(16, 9))
    for img_path in image_paths:
        if os.path.isfile(img_path):
            image = Image.open(img_path)

            plt.subplot(2, 3, images_shown + 1)
            plt.imshow(image)
            plt.xticks([])
            plt.yticks([])

            images_shown += 1
            if images_shown >= 9:
                break

def download_wiki_images(data_path):
    # Create the folder if it does not exist, using exist_ok=True to avoid errors if it already exists
    data_path.mkdir(exist_ok=True)

    image_uuid = 0
    # image_metadata_dict stores images metadata including image uuid, filename, and path
    image_metadata_dict = {}
    MAX_IMAGES_PER_WIKI = 30

    wiki_titles = [
        "San Francisco",
        "Batman",
        "Vincent van Gogh",
        "iPhone",
        "Tesla Model S",
        "BTS band",
    ]

    # Download images for wiki pages and assign UUID for each image
    for title in wiki_titles:
        images_per_wiki = 0
        print(title)
        try:
            page_py = wikipedia.page(title)
            list_img_urls = page_py.images
            for url in list_img_urls:
                if url.endswith(".jpg") or url.endswith(".png"):
                    image_uuid += 1
                    image_file_name = title + "_" + url.split("/")[-1]

                    # img_path could be s3 path pointing to the raw image file in the future
                    image_metadata_dict[image_uuid] = {
                        "filename": image_file_name,
                        "img_path": str(data_path / f"{image_uuid}.jpg"),
                    }
                    urllib.request.urlretrieve(
                        url, data_path / f"{image_uuid}.jpg"
                    )
                    images_per_wiki += 1
                    # Limit the number of images downloaded per wiki page to 15
                    if images_per_wiki > MAX_IMAGES_PER_WIKI:
                        break
        except Exception as e:
            print(f"No images found for Wikipedia page: {title}. Error: {e}")
            continue

    return image_metadata_dict
