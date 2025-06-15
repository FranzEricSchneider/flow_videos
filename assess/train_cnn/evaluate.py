"""Script to evaluate trained CNN models on new images."""

import argparse
import json
from pathlib import Path
import numpy
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from typing import List, Dict, Any, Tuple

from assess.train_cnn.models import MODELS
from assess.train_cnn.dataset import ImageRegressionDataset, get_train_test_datasets


def load_model_from_dir(model_dir: Path, device: torch.device) -> torch.nn.Module:
    """Load a trained model from a directory containing model weights and metadata.
    
    Args:
        model_dir: Directory containing best_model.pt and metadata.json
        device: Device to load model on
        
    Returns:
        Loaded model
    """
    # Load metadata to get model key
    metadata = json.load((model_dir / "metadata.json").open("r"))
    model_key = metadata["kwargs"]["model_key"]
    
    # Load model
    model = MODELS[model_key]().to(device)
    model.load_state_dict(torch.load(model_dir / "best_model.pt"))
    model.eval()
    return model


def evaluate_model(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: torch.device,
) -> numpy.ndarray:
    """Run inference on a dataset.
    
    Args:
        model: Trained model
        dataloader: DataLoader containing images
        device: Device to run inference on
        
    Returns:
        Array of predictions
    """
    all_preds = []
    
    with torch.no_grad():
        for images, _ in tqdm(dataloader, desc="Evaluating"):
            images = images.to(device)
            outputs = model(images)
            all_preds.extend(outputs.cpu().numpy())
            
    return numpy.array(all_preds)


def evaluate_ensemble(
    models: List[torch.nn.Module],
    dataloader: DataLoader,
    device: torch.device,
) -> numpy.ndarray:
    """Run inference using an ensemble of models.
    
    Args:
        models: List of trained models
        dataloader: DataLoader containing images
        device: Device to run inference on
        
    Returns:
        Array of ensemble predictions
    """
    all_preds = []
    
    with torch.no_grad():
        for images, _ in tqdm(dataloader, desc="Evaluating ensemble"):
            images = images.to(device)
            # Get predictions from each model
            model_preds = []
            for model in models:
                outputs = model(images)
                model_preds.append(outputs.cpu().numpy())
            # Average predictions
            ensemble_preds = numpy.mean(model_preds, axis=0)
            all_preds.extend(ensemble_preds)
            
    return numpy.array(all_preds)


def calculate_metrics(
    labels: numpy.ndarray, predictions: numpy.ndarray
) -> Dict[str, float]:
    """Calculate regression metrics.
    
    Args:
        labels: True labels
        predictions: Model predictions
        
    Returns:
        Dictionary of metrics
    """
    return {
        "r2": float(r2_score(labels, predictions)),
        "rmse": float(numpy.sqrt(mean_squared_error(labels, predictions))),
        "mae": float(mean_absolute_error(labels, predictions)),
    }


def evaluate_labeled_data(
    models: List[torch.nn.Module],
    data_dir: Path,
    device: torch.device,
    batch_size: int,
) -> Dict[str, Any]:
    """Evaluate models on labeled data.
    
    Args:
        models: List of trained models
        data_dir: Directory containing metadata.json and images
        device: Device to run evaluation on
        batch_size: Batch size for evaluation
        
    Returns:
        Dictionary of results
    """
    # Load metadata and create datasets
    metadata = json.load((data_dir / "metadata.json").open("r"))
    train_dataset, test_dataset = get_train_test_datasets(
        metadata=metadata,
        data_dir=data_dir,
        split_key="even",
        state_key="mass (g)",
    )
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    
    # Get predictions
    train_preds = evaluate_ensemble(models, train_loader, device)
    test_preds = evaluate_ensemble(models, test_loader, device)
    
    # Get true labels
    train_labels = train_dataset.labels.numpy()
    test_labels = test_dataset.labels.numpy()
    
    # Get image paths
    train_paths = [
        str(Path(p).relative_to(data_dir)) for p in train_dataset.image_paths
    ]
    test_paths = [
        str(Path(p).relative_to(data_dir)) for p in test_dataset.image_paths
    ]
    
    return {
        "train": {
            "predictions": train_preds.tolist(),
            "labels": train_labels.tolist(),
            "paths": train_paths,
            "metrics": calculate_metrics(train_labels, train_preds),
        },
        "test": {
            "predictions": test_preds.tolist(),
            "labels": test_labels.tolist(),
            "paths": test_paths,
            "metrics": calculate_metrics(test_labels, test_preds),
        },
    }


def evaluate_unlabeled_data(
    models: List[torch.nn.Module],
    data_dir: Path,
    device: torch.device,
    batch_size: int,
) -> Dict[str, Any]:
    """Evaluate models on unlabeled data.
    
    Args:
        models: List of trained models
        data_dir: Directory containing images
        device: Device to run evaluation on
        batch_size: Batch size for evaluation
        
    Returns:
        Dictionary of results
    """
    # Find images
    image_files = list(data_dir.glob("*.jpg")) + list(
        data_dir.glob("*.png")
    )
    if not image_files:
        raise ValueError(f"No images found in {data_dir}")
        
    # Create dataset with dummy labels
    dataset = ImageRegressionDataset(
        image_paths=[str(p) for p in image_files],
        labels=numpy.zeros(len(image_files)),
        is_train=False,
    )
    
    # Create dataloader
    dataloader = DataLoader(dataset, batch_size=batch_size)
    
    # Get predictions
    predictions = evaluate_ensemble(models, dataloader, device)
    
    # Get relative paths
    paths = [str(p.relative_to(data_dir)) for p in image_files]
    
    return {"predictions": predictions.tolist(), "paths": paths}


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "data_dir",
        type=Path,
        help="Directory containing images or metadata.json with train/test split",
    )
    parser.add_argument(
        "save_dir",
        type=Path,
        help="Directory to save results",
    )
    parser.add_argument(
        "--model-dirs",
        type=Path,
        nargs="+",
        required=True,
        help="Directories containing best_model.pt and metadata.json",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for evaluation",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run evaluation on",
    )
    
    return parser.parse_args()


def main() -> None:
    """Main evaluation function."""
    args = parse_args()
    
    # Create save directory
    args.save_dir.mkdir(parents=True, exist_ok=True)
    
    # Load models
    device = torch.device(args.device)
    models = [load_model_from_dir(d, device) for d in args.model_dirs]
    
    # Store arguments and model info in results
    results = {
        "args": {
            "model_dirs": [str(d) for d in args.model_dirs],
            "batch_size": args.batch_size,
            "device": args.device,
        },
        "models": [
            {
                "dir": str(d),
                "model_key": json.load((d / "metadata.json").open("r"))["kwargs"]["model_key"]
            }
            for d in args.model_dirs
        ]
    }
    
    # Evaluate based on data type
    if (args.data_dir / "metadata.json").exists():
        results.update(evaluate_labeled_data(models, args.data_dir, device, args.batch_size))
    else:
        results.update(evaluate_unlabeled_data(models, args.data_dir, device, args.batch_size))
    
    # Save results
    json.dump(
        results,
        (args.save_dir / "evaluation_results.json").open("w"),
        indent=2,
        sort_keys=True,
    )


if __name__ == "__main__":
    main()
