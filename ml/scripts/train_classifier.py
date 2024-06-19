import os
import argparse
import time
import joblib

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler

from torchvision import models
from torchvision.transforms import v2
from torchvision.models import ResNet50_Weights, EfficientNet_V2_S_Weights

from augment import RandAugmentMod
from utils import load_images_to_dataframe, ImageFolderSubset


def generate_datasets(base_path: str, raw_df: pd.DataFrame, n: int):

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

    label_encoder = LabelEncoder()
    label_encoder.fit_transform(df["label"])
    labels = label_encoder.classes_


    return train_df, valid_df, test_df, labels


if __name__ == "__main__":

    argparser = argparse.ArgumentParser()

    argparser.add_argument("--data", type=str, help="Root path of the dataset")
    argparser.add_argument("--n", type=int, help="Choose n most common species to classify")
    argparser.add_argument("--epochs", type=int, required=False, help="Number of epochs to train the model")
    argparser.add_argument("--checkpoint", type=str, help="File path to checkpoint to resume training from")
    argparser.add_argument("--timeout", type=int, help="Terminate training after this many hours")

    args = argparser.parse_args()

    base_path = args.data
    n = args.n
    num_epochs = args.epochs or 100
    chkpt_path = args.checkpoint

    weights = EfficientNet_V2_S_Weights.IMAGENET1K_V1
    model = models.efficientnet_v2_s(weights=weights)
    raw_df, data = load_images_to_dataframe(base_path)

    save_path = chkpt_path or ("sessions/" + time.strftime("%Y%m%d-%H%M"))
    os.makedirs(save_path, exist_ok=True)

    transform = weights.transforms()

    train_transform = v2.Compose(
        [
            v2.ToDtype(torch.uint8),
            RandAugmentMod(),
            v2.ToDtype(torch.float32),
            transform,
        ]
    )

    if chkpt_path:
        train_df = joblib.load(f"{save_path}/trainset.joblib")
        valid_df = joblib.load(f"{save_path}/validset.joblib")
        test_df = joblib.load(f"{save_path}/testset.joblib")
        labels = joblib.load(f"{save_path}/labels.joblib")

    else:

        train_df, valid_df, test_df, labels = generate_datasets(base_path, raw_df, n)

        # pickle and save the datasets
        joblib.dump(train_df, f"{save_path}/trainset.joblib")
        joblib.dump(valid_df, f"{save_path}/validset.joblib")
        joblib.dump(test_df, f"{save_path}/testset.joblib")
        joblib.dump(labels, f"{save_path}/labels.joblib")

    label_encoder = LabelEncoder()
    label_encoder.fit(labels)

    trainset = ImageFolderSubset(
        base_path,
        files=train_df["path"].tolist(),
        transform=train_transform,
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

    num_classes = len(labels)
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

    time_limit = args.timeout * 3600 if args.timeout else float("inf")

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
                    f"{save_path}/best_checkpoint.pth",
                )

                joblib.dump({
                    "model": model,
                    "labels": labels,
                    "transform": transform
                    },
                    f"{save_path}/model.joblib"
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
