import os
import argparse
import sys
import requests
import pathlib
from typing import Optional, Callable
import json
from functools import partial
import time

import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models
from torchvision.transforms import v2
from torchvision.models import ResNet50_Weights, EfficientNet_V2_S_Weights
from torchvision.datasets import ImageFolder, VisionDataset
from torch.cuda.amp import autocast, GradScaler


def load_images_to_dataframe(base_path):
    data_indices = []
    labels = []
    paths = []

    data = {}

    # List all subdirectories in the base path
    for i, species_name in enumerate(os.listdir(base_path)):
        species_path = os.path.join(base_path, species_name)

        # Check if the path is a directory
        if os.path.isdir(species_path):
            for i, image_name in enumerate(os.listdir(species_path)):
                # take at most 500 images per class
                # if i >= 500:
                #    break

                is_image = any(
                    image_name.endswith(ext) for ext in [".jpg", ".jpeg", ".png"]
                )

                if is_image:
                    image_path = os.path.join(species_path, image_name)
                    try:
                        # Open the image, convert to RGB and then to a NumPy array
                        # image = Image.open(image_path).convert('RGB')

                        # Store the image data and the label
                        index = f"{species_name}_{i}"
                        data_indices.append(index)
                        # data[index] = image
                        labels.append(species_name)
                        paths.append(image_path)
                    except Exception as e:
                        print(f"Error loading {image_path}: {e}")

    # Create a DataFrame
    df = pd.DataFrame({"data_idx": data_indices, "label": labels, "path": paths})

    return df, data


def scientific_to_vernaculars(name: str) -> list[str]:
    response = requests.get(f"https://api.gbif.org/v1/species/search?q={name}")
    data = response.json()
    ret = []
    if data["results"]:
        for result in data["results"]:
            names = [obj["vernacularName"] for obj in result["vernacularNames"]]
            ret += names
    return ret


def vernacular_to_scientific(name: str) -> Optional[str]:
    response = requests.get(f"https://api.gbif.org/v1/species/search?q={name}")
    data = response.json()
    if data["results"]:
        return data["results"][0].get("canonicalName")


class ImageFolderSubset(VisionDataset):
    def __init__(
        self,
        root: str,
        files: list[str],
        transform: Callable,
        target_transform: Callable,
    ):
        super().__init__(root, transform=transform, target_transform=target_transform)
        self.files = files

    def __getitem__(self, index):
        path = self.files[index]
        # take first part of relative path after root as target
        target = pathlib.Path(path).relative_to(self.root).parts[0]
        target = self.target_transform(target)
        img = Image.open(path).convert("RGB")
        img = self.transform(img)
        return img, target

    def __len__(self):
        return len(self.files)


if __name__ == "__main__":

    argparser = argparse.ArgumentParser()

    argparser.add_argument("path", type=str, help="Path to model to evaluate")

    args = argparser.parse_args()

    path = args.path

    best_model_params = torch.load(os.path.join(path, "best_model.pth"))

    with open(os.path.join(path, "labels.json")) as f:
        testset_raw = json.load(f)

    test_df = load_images_to_dataframe(testset_raw)
    best_model = models.efficientnet_v2_s()

    num_classes = test_df["label"].nunique()
    best_model.classifier[1] = nn.Linear(best_model.classifier[1].in_features, num_classes)
    best_model.load_state_dict(best_model_params["model"])

    transform = EfficientNet_V2_S_Weights.transforms()

    label_encoder = LabelEncoder()
    label_encoder.fit_transform(test_df["label"])

    testset = ImageFolderSubset(
        path,
        files=test_df["path"].tolist(),
        transform=transform,
        target_transform=lambda x: label_encoder.transform([x])[0],
    )

    test_loader = DataLoader(testset, batch_size=8, shuffle=False)

    best_accuracy = 0
    best_model.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    pred_targets = []

            # Validation Step
    with torch.no_grad():
        val_loss = 0
        correct = 0
        total = 0
        for data in test_loader:
            inputs, labels = data[0].to(device), data[1].to(device)
            outputs = best_model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
            pred_targets += zip(predicted.tolist(), labels.tolist())
    test_accuracy = 100 * correct / total
    print(
        f"Test accuracy: {test_accuracy:.2f}%"
    )

    # plot confusion matrix
    from sklearn.metrics import confusion_matrix
    import seaborn as sns

    pred, target = zip(*pred_targets)
    cm = confusion_matrix(target, pred)
    plt.figure(figsize=(10, 10))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()


