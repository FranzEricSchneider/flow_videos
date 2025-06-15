"""Dataset utilities for CNN training."""

import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path
from PIL import Image
import torchvision.transforms as transforms
from assess.train_classic.train import get_labels
from assess.train_classic.allocate import ALLOCATE


class ImageRegressionDataset(Dataset):
    """Dataset for image regression tasks."""

    def __init__(self, image_paths: list[str], labels: np.ndarray, transform=None):
        """Initialize dataset.

        Args:
            image_paths: List of paths to images
            labels: Array of regression target values
            transform: Optional transform to apply to images
        """
        self.image_paths = image_paths
        self.labels = torch.tensor(labels, dtype=torch.float32)

        if transform is None:
            # Default transforms for training
            self.transform = transforms.Compose(
                [
                    transforms.Resize((224, 224)),  # Standard input size for many CNNs
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.485, 0.456, 0.406],  # ImageNet stats
                        std=[0.229, 0.224, 0.225],
                    ),
                ]
            )
        else:
            self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert("RGB")
        label = self.labels[idx]
        return self.transform(image), label


def get_train_test_datasets(
    metadata: dict, data_dir: Path, split_key: str, state_key: str
) -> tuple[Dataset, Dataset]:
    """Create train and test datasets from metadata.

    Args:
        metadata: Metadata dictionary containing image paths and labels
        data_dir: Base directory containing the images
        split_key: Key for how to split state across frames
        state_key: Key to look up state we want to regress to

    Returns:
        Tuple of (train_dataset, test_dataset)
    """
    # Get train and test image paths
    train_paths = []
    test_paths = []

    for img_path in metadata["images"].keys():
        full_path = data_dir / img_path
        if "train" in str(full_path):
            train_paths.append(img_path)
        elif "test" in str(full_path):
            test_paths.append(img_path)

    # Get labels using the same method as classic training
    splitter = ALLOCATE[split_key]
    train_labels = get_labels(splitter, metadata, state_key, train_paths)
    test_labels = get_labels(splitter, metadata, state_key, test_paths)

    # Create full paths for images
    train_full_paths = [str(data_dir / path) for path in train_paths]
    test_full_paths = [str(data_dir / path) for path in test_paths]

    return (
        ImageRegressionDataset(train_full_paths, train_labels),
        ImageRegressionDataset(test_full_paths, test_labels),
    )
