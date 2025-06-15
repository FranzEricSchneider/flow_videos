"""Script to visualize training augmentations by saving augmented images."""

# Standard library imports
import argparse
import shutil
from pathlib import Path
from typing import List

# Third-party imports
import numpy
import torch
from PIL import Image
from tqdm import tqdm

# Local imports
from assess.train_cnn.dataset import ImageRegressionDataset


def save_augmented_images(data_dir: Path, save_dir: Path, num_samples: int = 5) -> None:
    """Save augmented versions of images for visualization.

    Args:
        data_dir: Directory containing input images
        save_dir: Directory to save augmented images
        num_samples: Number of augmented versions to create per image

    Raises:
        ValueError: If no images are found in data_dir
        OSError: If there are issues with file operations
    """
    # Create save directory
    save_dir.mkdir(parents=True, exist_ok=True)

    # Get list of image files
    image_files: List[Path] = list(data_dir.glob("*.jpg")) + list(
        data_dir.glob("*.png")
    )
    if not image_files:
        raise ValueError(f"No images found in {data_dir}")

    # Create dummy labels (not used for visualization)
    dummy_labels = numpy.zeros(len(image_files))

    # Create dataset with training augmentations
    dataset = ImageRegressionDataset(
        image_paths=[str(p) for p in image_files],
        labels=dummy_labels,
        is_train=True,
        normalize=False,  # Turn off normalization for visualization
    )

    # Process each image with progress bar
    for img_idx, img_path in enumerate(tqdm(image_files, desc="Processing images")):
        # Save original image
        orig_save_path = save_dir / f"{img_path.stem}_original{img_path.suffix}"
        shutil.copy2(img_path, orig_save_path)

        # Generate and save augmented versions
        for aug_idx in range(num_samples):

            # Convert tensor to PIL Image and save
            aug_img, _ = dataset[img_idx]
            aug_img = aug_img.permute(1, 2, 0).numpy()  # CHW -> HWC
            aug_img = Image.fromarray(aug_img)

            aug_save_path = save_dir / f"{img_path.stem}_aug{aug_idx}{img_path.suffix}"
            aug_img.save(aug_save_path)


def main() -> None:

    parser = argparse.ArgumentParser(description="Visualize training augmentations")
    parser.add_argument("data_dir", type=Path, help="Directory containing input images")
    parser.add_argument(
        "save_dir", type=Path, help="Directory to save augmented images"
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=5,
        help="Number of augmented versions to create per image",
    )

    args = parser.parse_args()
    save_augmented_images(args.data_dir, args.save_dir, args.num_samples)


if __name__ == "__main__":
    main()
