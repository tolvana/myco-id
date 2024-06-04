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

    argparser.add_argument("--data", type=str, help="Root path of the dataset")
    argparser.add_argument("--n", type=int, help="Choose n most common species to classify")
    argparser.add_argument("--num_epochs", type=int, help="Number of epochs to train the model")

    args = argparser.parse_args()

    base_path = args.data
    n = args.n
    num_epochs = args.num_epochs
    print("args:", args)

    weights = EfficientNet_V2_S_Weights.IMAGENET1K_V1
    model = models.efficientnet_v2_s(weights=weights)
    raw_df, data = load_images_to_dataframe(base_path)

    print(f"Loaded {len(raw_df)} samples")

    # choose only n most common species
    species_counts = raw_df["label"].value_counts()

    top_species = species_counts.head(n)
    print(f"Top {n} species: {top_species}")
    raw_df = raw_df[raw_df["label"].isin(top_species.index)]

    # make sure dataset is balanced. If not, subsample the larger classes
    minimum_count = top_species.min()
    print(f"Minimum count: {minimum_count}")
    df = pd.concat(
        [
            raw_df[raw_df["label"] == label].sample(minimum_count)
            for label in top_species.index
        ]
    )
    print(f"balanced dataset: {len(df)} samples out of {len(raw_df)}")

    train_df, tmp_df = train_test_split(
        df, test_size=0.1, random_state=42, stratify=df["label"]
    )
    valid_df, test_df = train_test_split(
        tmp_df, test_size=0.5, random_state=42, stratify=tmp_df["label"]
    )

    # date and time as a save path
    save_path = "sessions/" + time.strftime("%Y%m%d-%H%M%S")
    os.makedirs(save_path, exist_ok=True)

    # save list of testset paths and labels to disk
    print(f"saving {len(test_df)} test samples to {save_path}/testset.json")
    with open(save_path + "/testset.json", "w") as f:
        json.dump(
            {"paths": list(test_df["path"]), "labels": list(test_df["label"])},
            f,
        )

    label_encoder = LabelEncoder()
    label_encoder.fit_transform(df["label"])
    labels = label_encoder.classes_
    with open(f"{save_path}/labels.json", "w") as f:
        json.dump(list(labels), f)

    transform = weights.transforms()

    train_transform = v2.Compose(
        [
            v2.ToDtype(torch.uint8),
            v2.RandAugment(),
            v2.ToDtype(torch.float32),
            transform,
        ]
    )

    trainset = ImageFolderSubset(
        base_path,
        files=train_df["path"].tolist(),
        transform=transform,
        target_transform=lambda x: label_encoder.transform([x])[0],
    )

    # testset = ImageFolder(
    #    base_path,
    #    transform=transform,
    #    target_transform=label_encoder.transform,
    #    is_valid_file=partial(is_valid_file, set(test_df["path"]))
    # )

    validset = ImageFolderSubset(
        base_path,
        files=valid_df["path"].tolist(),
        transform=transform,
        target_transform=lambda x: label_encoder.transform([x])[0],
    )

    train_loader = DataLoader(trainset, batch_size=32, shuffle=True, num_workers=16, pin_memory=True)
    # test_loader = DataLoader(testset, batch_size=32, shuffle=False)
    valid_loader = DataLoader(validset, batch_size=32, shuffle=False, num_workers=16, pin_memory=True)

    num_classes = len(df["label"].unique())
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)

    # Loss Function and Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # cosine annealing with warm restarts
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=3, T_mult=2
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device, memory_format=torch.channels_last)
    time_limit = 10 * 3600 # 10 hours

    scaler = GradScaler()

    best_accuracy = 0
    t0 = time.time()
    try:
        for epoch in range(num_epochs):
            # Training Step
            model.train()
            t0 = time.time()
            for i, data in enumerate(train_loader):
                print(
                    f"\rEpoch {epoch + 1}/{num_epochs}, Batch {i+1}/{len(train_loader)}",
                    end="",
                    flush=True,
                )
                inputs, labels = data[0].to(device, memory_format=torch.channels_last), data[1].to(device)
                optimizer.zero_grad()
                with autocast():
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

                t0 = time.time()

            scheduler.step()

            # Validation Step
            model.eval()
            with torch.no_grad():
                val_loss = 0
                correct = 0
                total = 0
                for data in valid_loader:
                    inputs, labels = data[0].to(device), data[1].to(device)
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    val_loss += loss.item()
                    _, predicted = torch.max(outputs.data, 1)
                    correct += (predicted == labels).sum().item()
                    total += labels.size(0)
                validation_accuracy = 100 * correct / total
                print(
                    f"\rEpoch {epoch + 1}/{num_epochs}, "
                    f"Loss: {val_loss:.2f}, "
                    f"Validation Accuracy: {validation_accuracy:.2f}%"
                )
            if validation_accuracy > best_accuracy:
                best_accuracy = validation_accuracy
                print("Saving best model")
                torch.save(
                    {
                        "model": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "scaler": scaler.state_dict(),
                    },
                    f"{save_path}/best_model.pth",
                )

            # save model checkpoint
            torch.save(
                {
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scaler": scaler.state_dict(),
                },
                f"{save_path}/checkpoint.pth",
            )
            if time.time() - t0 > time_limit:
                print("Time limit reached")
                break

    except KeyboardInterrupt:
        pass
