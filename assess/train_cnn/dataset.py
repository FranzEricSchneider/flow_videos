"""Dataset utilities for CNN training."""

# Standard library imports
from pathlib import Path

# Third-party imports
import albumentations
from albumentations.pytorch import ToTensorV2
from PIL import Image
import numpy as np
import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset

# Local imports
from assess.train_classic.allocate import ALLOCATE
from assess.train_classic.train import get_labels


# ImageNet normalization parameters
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# Standard image size for most CNN architectures
IMAGE_SIZE = 224


class ImageRegressionDataset(Dataset):
    """Dataset for image regression tasks."""

    def __init__(
        self,
        image_paths: list[str],
        labels: np.ndarray,
        transform=None,
        is_train: bool = False,
        normalize: bool = True,
    ):
        """Initialize dataset.

        Args:
            image_paths: List of paths to images
            labels: Array of regression target values
            transform: Optional transform to apply to images
            is_train: Whether this is the training set (for augmentations)
            normalize: Whether to apply ImageNet normalization
        """
        self.image_paths = image_paths
        self.labels = torch.tensor(labels, dtype=torch.float32)
        self.is_train = is_train

        if transform is None:
            if is_train:
                # Training augmentations
                transforms_list = [
                    # Geometric transforms
                    albumentations.HorizontalFlip(p=0.3),
                    albumentations.VerticalFlip(p=0.3),
                    albumentations.Affine(
                        translate_percent=0.05,
                        scale=(0.95, 1.05),
                        rotate=(-5, 5),
                        p=0.5,
                    ),
                    # Color transforms
                    albumentations.OneOf(
                        [
                            albumentations.RandomBrightnessContrast(
                                brightness_limit=0.1, contrast_limit=0.1, p=1
                            ),
                            albumentations.RandomGamma(gamma_limit=(90, 110), p=1),
                        ],
                        p=0.3,
                    ),
                    # Noise and blur
                    albumentations.OneOf(
                        [
                            albumentations.GaussNoise(var_limit=(0.01, 0.3), p=1),
                            albumentations.GaussianBlur(blur_limit=(3, 5), p=1),
                        ],
                        p=0.3,
                    ),
                    # Resize
                    albumentations.Resize(IMAGE_SIZE, IMAGE_SIZE),
                ]

                if normalize:
                    transforms_list.extend(
                        [
                            albumentations.Normalize(
                                mean=IMAGENET_MEAN, std=IMAGENET_STD
                            ),
                        ]
                    )

                transforms_list.append(ToTensorV2())

                self.transform = albumentations.Compose(transforms_list)
            else:
                # Validation/test transforms
                transforms_list = [
                    albumentations.Resize(IMAGE_SIZE, IMAGE_SIZE),
                ]

                if normalize:
                    transforms_list.extend(
                        [
                            albumentations.Normalize(
                                mean=IMAGENET_MEAN, std=IMAGENET_STD
                            ),
                        ]
                    )

                transforms_list.append(ToTensorV2())

                self.transform = albumentations.Compose(transforms_list)
        else:
            self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert("RGB")
        image = np.array(image)  # Convert PIL Image to numpy array
        label = self.labels[idx]

        if self.transform:
            transformed = self.transform(image=image)
            image = transformed["image"]

        return image, label


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
        ImageRegressionDataset(train_full_paths, train_labels, is_train=True),
        ImageRegressionDataset(test_full_paths, test_labels, is_train=False),
    )
