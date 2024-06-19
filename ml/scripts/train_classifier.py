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

from torchvision.transforms import functional as F, InterpolationMode
from torch import Tensor
from enum import Enum
from typing import Dict, List, Optional, Tuple

def _apply_op(
    img: Tensor, op_name: str, magnitude: float, interpolation: InterpolationMode, fill: Optional[List[float]]
):
    if op_name == "ShearX":
        # magnitude should be arctan(magnitude)
        # official autoaug: (1, level, 0, 0, 1, 0)
        # https://github.com/tensorflow/models/blob/dd02069717128186b88afa8d857ce57d17957f03/research/autoaugment/augmentation_transforms.py#L290
        # compared to
        # torchvision:      (1, tan(level), 0, 0, 1, 0)
        # https://github.com/pytorch/vision/blob/0c2373d0bba3499e95776e7936e207d8a1676e65/torchvision/transforms/functional.py#L976
        img = F.affine(
            img,
            angle=0.0,
            translate=[0, 0],
            scale=1.0,
            shear=[math.degrees(math.atan(magnitude)), 0.0],
            interpolation=interpolation,
            fill=fill,
            center=[0, 0],
        )
    elif op_name == "ShearY":
        # magnitude should be arctan(magnitude)
        # See above
        img = F.affine(
            img,
            angle=0.0,
            translate=[0, 0],
            scale=1.0,
            shear=[0.0, math.degrees(math.atan(magnitude))],
            interpolation=interpolation,
            fill=fill,
            center=[0, 0],
        )
    elif op_name == "TranslateX":
        img = F.affine(
            img,
            angle=0.0,
            translate=[int(magnitude), 0],
            scale=1.0,
            interpolation=interpolation,
            shear=[0.0, 0.0],
            fill=fill,
        )
    elif op_name == "TranslateY":
        img = F.affine(
            img,
            angle=0.0,
            translate=[0, int(magnitude)],
            scale=1.0,
            interpolation=interpolation,
            shear=[0.0, 0.0],
            fill=fill,
        )
    elif op_name == "Rotate":
        img = F.rotate(img, magnitude, interpolation=interpolation, fill=fill)
    elif op_name == "Brightness":
        img = F.adjust_brightness(img, 1.0 + magnitude)
    elif op_name == "Color":
        img = F.adjust_saturation(img, 1.0 + magnitude)
    elif op_name == "Contrast":
        img = F.adjust_contrast(img, 1.0 + magnitude)
    elif op_name == "Sharpness":
        img = F.adjust_sharpness(img, 1.0 + magnitude)
    elif op_name == "Posterize":
        img = F.posterize(img, int(magnitude))
    elif op_name == "Solarize":
        img = F.solarize(img, magnitude)
    elif op_name == "AutoContrast":
        img = F.autocontrast(img)
    elif op_name == "Equalize":
        img = F.equalize(img)
    elif op_name == "Invert":
        img = F.invert(img)
    elif op_name == "Identity":
        pass
    else:
        raise ValueError(f"The provided operator {op_name} is not recognized.")
    return img

class RandAugmentMod(torch.nn.Module):
    r"""RandAugment data augmentation method based on
    `"RandAugment: Practical automated data augmentation with a reduced search space"
    <https://arxiv.org/abs/1909.13719>`_.
    If the image is torch Tensor, it should be of type torch.uint8, and it is expected
    to have [..., 1 or 3, H, W] shape, where ... means an arbitrary number of leading dimensions.
    If img is PIL Image, it is expected to be in mode "L" or "RGB".

    Args:
        num_ops (int): Number of augmentation transformations to apply sequentially.
        magnitude (int): Magnitude for all the transformations.
        num_magnitude_bins (int): The number of different magnitude values.
        interpolation (InterpolationMode): Desired interpolation enum defined by
            :class:`torchvision.transforms.InterpolationMode`. Default is ``InterpolationMode.NEAREST``.
            If input is Tensor, only ``InterpolationMode.NEAREST``, ``InterpolationMode.BILINEAR`` are supported.
        fill (sequence or number, optional): Pixel fill value for the area outside the transformed
            image. If given a number, the value is used for all bands respectively.
    """

    def __init__(
        self,
        num_ops: int = 2,
        magnitude: int = 9,
        num_magnitude_bins: int = 31,
        interpolation: InterpolationMode = InterpolationMode.NEAREST,
        fill: Optional[List[float]] = None,
    ) -> None:
        super().__init__()
        self.num_ops = num_ops
        self.magnitude = magnitude
        self.num_magnitude_bins = num_magnitude_bins
        self.interpolation = interpolation
        self.fill = fill

    def _augmentation_space(self, num_bins: int, image_size: Tuple[int, int]) -> Dict[str, Tuple[Tensor, bool]]:
        return {
            # op_name: (magnitudes, signed)
            "Identity": (torch.tensor(0.0), False),
            "ShearX": (torch.linspace(0.0, 0.3, num_bins), True),
            "ShearY": (torch.linspace(0.0, 0.3, num_bins), True),
            "TranslateX": (torch.linspace(0.0, 150.0 / 331.0 * image_size[1], num_bins), True),
            "TranslateY": (torch.linspace(0.0, 150.0 / 331.0 * image_size[0], num_bins), True),
            "Rotate": (torch.linspace(0.0, 30.0, num_bins), True),
            "Brightness": (torch.linspace(0.0, 0.9, num_bins), True),
            "Color": (torch.linspace(0.0, 0.9, num_bins), True),
            "Contrast": (torch.linspace(0.0, 0.9, num_bins), True),
            "Sharpness": (torch.linspace(0.0, 0.9, num_bins), True),
            "Posterize": (8 - (torch.arange(num_bins) / ((num_bins - 1) / 4)).round().int(), False),
            #"Solarize": (torch.linspace(255.0, 0.0, num_bins), False),
            "AutoContrast": (torch.tensor(0.0), False),
            "Equalize": (torch.tensor(0.0), False),
        }

    def forward(self, img: Tensor) -> Tensor:
        """
            img (PIL Image or Tensor): Image to be transformed.

        Returns:
            PIL Image or Tensor: Transformed image.
        """
        fill = self.fill
        channels, height, width = F.get_dimensions(img)
        if isinstance(img, Tensor):
            if isinstance(fill, (int, float)):
                fill = [float(fill)] * channels
            elif fill is not None:
                fill = [float(f) for f in fill]

        op_meta = self._augmentation_space(self.num_magnitude_bins, (height, width))
        for _ in range(self.num_ops):
            op_index = int(torch.randint(len(op_meta), (1,)).item())
            op_name = list(op_meta.keys())[op_index]
            magnitudes, signed = op_meta[op_name]
            magnitude = float(magnitudes[self.magnitude].item()) if magnitudes.ndim > 0 else 0.0
            if signed and torch.randint(2, (1,)):
                magnitude *= -1.0
            img = _apply_op(img, op_name, magnitude, interpolation=self.interpolation, fill=fill)

        return img


    def __repr__(self) -> str:
        s = (
            f"{self.__class__.__name__}("
            f"num_ops={self.num_ops}"
            f", magnitude={self.magnitude}"
            f", num_magnitude_bins={self.num_magnitude_bins}"
            f", interpolation={self.interpolation}"
            f", fill={self.fill}"
            f")"
        )
        return s

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
    argparser.add_argument("--checkpoint", type=str, help="File path to checkpoint to resume training from")

    args = argparser.parse_args()

    base_path = args.data
    n = args.n
    num_epochs = args.num_epochs
    chkpt_path = args.checkpoint
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
    # add discarded samples back to the train set
    train_df = raw_df[~raw_df["data_idx"].isin(test_df["data_idx"])]
    train_df = train_df[~train_df["data_idx"].isin(valid_df["data_idx"])]

    # make sure trainset is balanced. If not, repeat the smaller classes
    target_count = int(top_species.mean())
    print(f"Target count: {target_count}")
    df = pd.concat(
        [
            train_df[train_df["label"] == label].sample(target_count, replace=True)
            for label in top_species.index
        ]
    )

    print(f"{len(df)}")

    # date and time as a save path
    save_path = "sessions/" + time.strftime("%Y%m%d-%H%M")
    os.makedirs(save_path, exist_ok=True)

    # save list of testset paths and labels to disk
    print(f"saving {len(test_df)} test samples to {save_path}/testset.json")
    with open(save_path + "/testset.json", "w") as f:
        json.dump(
            {"paths": list(test_df["path"]), "labels": list(test_df["label"])},
            f,
        )

    # save list of validset paths and labels to disk

    print(f"saving {len(valid_df)} validation samples to {save_path}/validset.json")
    with open(save_path + "/validset.json", "w") as f:
        json.dump(
            {"paths": list(valid_df["path"]), "labels": list(valid_df["label"])},
            f,
        )

    # save list of trainset paths and labels to disk
    print(f"saving {len(df)} training samples to {save_path}/trainset.json")
    with open(save_path + "/trainset.json", "w") as f:
        json.dump(
            {"paths": list(df["path"]), "labels": list(df["label"])},
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
            RandAugmentMod(),
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

    validset = ImageFolderSubset(
        base_path,
        files=valid_df["path"].tolist(),
        transform=transform,
        target_transform=lambda x: label_encoder.transform([x])[0],
    )

    train_loader = DataLoader(trainset, batch_size=32, shuffle=True, num_workers=12, pin_memory=True)
    valid_loader = DataLoader(validset, batch_size=32, shuffle=False, num_workers=12, pin_memory=True)

    num_classes = len(df["label"].unique())
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)

    # Loss Function and Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    scaler = GradScaler()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = model.to(device, memory_format=torch.channels_last)

    if chkpt_path:
        checkpoint = torch.load(chkpt_path)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scaler.load_state_dict(checkpoint["scaler"])

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, mode="min")

    time_limit = 36 * 3600 # seconds

    best_accuracy = 0
    t0 = time.time()
    try:
        for epoch in range(num_epochs):
            # Training Step
            model.train()
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

            scheduler.step(val_loss)

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
