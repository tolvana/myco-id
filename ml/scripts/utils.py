import os
import pandas as pd
import pathlib
from typing import Callable
from PIL import Image

from torchvision.datasets import VisionDataset


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

    def __getitem__(self, index: int):
        path = self.files[index]
        # take first part of relative path after root as target
        target = pathlib.Path(path).relative_to(self.root).parts[0]
        target = self.target_transform(target)
        img = Image.open(path).convert("RGB")
        img = self.transform(img)
        return img, target

    def __len__(self):
        return len(self.files)

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
