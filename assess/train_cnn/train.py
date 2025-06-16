"""Tool to train CNN-based regressors on image snippets."""

import argparse
import json
import numpy as np
from pathlib import Path
import time
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch.nn as nn
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import wandb

from assess.train_cnn.models import MODELS
from assess.train_cnn.dataset import get_train_test_datasets

# Load wandb key
with open(Path.home() / "wandb.json") as f:
    WANDB_KEY = json.load(f)["key"]
wandb.login(key=WANDB_KEY)


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    """Train model for one epoch.

    Args:
        model: CNN model
        dataloader: Training data loader
        criterion: Loss function
        optimizer: Optimizer
        device: Device to train on

    Returns:
        Average loss for the epoch
    """
    model.train()
    total_loss = 0.0

    pbar = tqdm(dataloader, desc="Training")  # , leave=False)
    for i, (images, labels) in enumerate(pbar):
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        pbar.set_postfix({"running loss": f"{total_loss / (i + 1):.3f}"})

    return total_loss / len(dataloader)


def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    label: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Evaluate model on dataloader.

    Args:
        model: CNN model
        dataloader: Data loader
        device: Device to evaluate on

    Returns:
        Tuple of (true labels, predicted labels)
    """
    model.eval()
    all_labels = []
    all_preds = []

    total_loss = 0.0
    criterion = nn.MSELoss()

    pbar = tqdm(dataloader, desc=f"Evaluating ({label})")  # , leave=False)
    with torch.no_grad():
        for i, (images, labels) in enumerate(pbar):
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)

            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(outputs.cpu().numpy())

            total_loss += criterion(outputs, labels).item()
            pbar.set_postfix({"running loss": f"{total_loss / (i + 1):.3f}"})

    return np.array(all_labels), np.array(all_preds)


def main(
    metadata: dict,
    data_dir: str,
    split_key: str,
    save_dir: str,
    state_key: str,
    model_key: str,
    batch_size: int,
    num_epochs: int,
    learning_rate: float,
    device: str,
    lite: bool,
):
    # Convert paths to pathlib
    data_dir = Path(data_dir)

    # Create output directory
    out_dir = Path(save_dir) / f"{data_dir.name}_{int(time.time() * 1e6)}"
    out_dir.mkdir()

    # Initialize wandb
    wandb.init(
        project="flow_videos",
        name=metadata["name"],
        config={
            "model": model_key,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "num_epochs": num_epochs,
            "split": split_key,
            "state_key": state_key,
        },
    )

    # Load previous step metadata
    previous = json.load(data_dir.joinpath("metadata.json").open("r"))

    # Create datasets and dataloaders
    train_dataset, test_dataset = get_train_test_datasets(
        metadata=previous, data_dir=data_dir, split_key=split_key, state_key=state_key
    )
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    # Initialize model and training components
    device = torch.device(device)
    model = MODELS[model_key]().to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Log model architecture
    wandb.watch(model, criterion, log="all", log_freq=5)

    # Training loop
    best_test_rmse = float("inf")
    for epoch in range(num_epochs):
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)

        # Evaluate on train and test sets
        train_labels, train_preds = evaluate(model, train_loader, device, "Train")
        test_labels, test_preds = evaluate(model, test_loader, device, "Test")

        # Calculate metrics
        train_r2 = r2_score(train_labels, train_preds)
        train_rmse = np.sqrt(mean_squared_error(train_labels, train_preds))
        train_mae = mean_absolute_error(train_labels, train_preds)

        test_r2 = r2_score(test_labels, test_preds)
        test_rmse = np.sqrt(mean_squared_error(test_labels, test_preds))
        test_mae = mean_absolute_error(test_labels, test_preds)

        # Log metrics to wandb
        wandb.log(
            {
                "epoch": epoch + 1,
                "train/loss": train_loss,
                "train/r2": train_r2,
                "train/rmse": train_rmse,
                "train/mae": train_mae,
                "test/r2": test_r2,
                "test/rmse": test_rmse,
                "test/mae": test_mae,
            }
        )

        print(f"Epoch {epoch+1}/{num_epochs}")
        print(
            f"Train - Loss: {train_loss:.4f}, R2: {train_r2:.4f}, RMSE: {train_rmse:.4f}, MAE: {train_mae:.4f}"
        )
        print(
            "Test  -"
            + " " * 15
            + f"R2: {test_r2:.4f}, RMSE: {test_rmse:.4f}, MAE: {test_mae:.4f}"
        )

        # Save best model
        if test_rmse < best_test_rmse:
            best_test_rmse = test_rmse
            torch.save(model.state_dict(), out_dir / "best_model.pt")
            wandb.save(str(out_dir / "best_model.pt"))

    # Load best model for final evaluation
    model.load_state_dict(torch.load(out_dir / "best_model.pt"))
    train_eval = evaluate(model, train_loader, device, "Train")
    test_eval = evaluate(model, test_loader, device, "Test")

    # Save results
    if not lite:
        metadata["y_train"] = train_eval[0].tolist()
        metadata["y_pred_train"] = train_eval[1].tolist()
        metadata["y_test"] = test_eval[0].tolist()
        metadata["y_pred_test"] = test_eval[1].tolist()

    # Save metrics
    metadata["R2_train"] = float(r2_score(*train_eval))
    metadata["RMSE_train"] = float(np.sqrt(mean_squared_error(*train_eval)))
    metadata["MAE_train"] = float(mean_absolute_error(*train_eval))
    metadata["R2_test"] = float(r2_score(*test_eval))
    metadata["RMSE_test"] = float(np.sqrt(mean_squared_error(*test_eval)))
    metadata["MAE_test"] = float(mean_absolute_error(*test_eval))

    # Save metadata
    json.dump(
        metadata,
        out_dir.joinpath("metadata.json").open("w"),
        indent=2,
        sort_keys=True,
    )

    # Log final metrics and close wandb
    wandb.log(
        {
            "final/train/r2": metadata["R2_train"],
            "final/train/rmse": metadata["RMSE_train"],
            "final/train/mae": metadata["MAE_train"],
            "final/test/r2": metadata["R2_test"],
            "final/test/rmse": metadata["RMSE_test"],
            "final/test/mae": metadata["MAE_test"],
        }
    )
    wandb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--data-dir",
        help="Path to processed data with metadata.json, train/, and test/.",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--save-dir",
        help="Directory in which to save the output model and metadata.",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--state-key",
        help="Key to look up state we want to regress to.",
        required=True,
    )
    parser.add_argument(
        "--split",
        help="Options for how to split the state across frames.",
        choices=["even"],  # Add more options as needed
        default="even",
    )
    parser.add_argument(
        "--model",
        help="Options for CNN models.",
        choices=sorted(MODELS.keys()),
        default="resnet18",
    )
    parser.add_argument(
        "--batch-size",
        help="Batch size for training.",
        type=int,
        default=32,
    )
    parser.add_argument(
        "--num-epochs",
        help="Number of training epochs.",
        type=int,
        default=50,
    )
    parser.add_argument(
        "--learning-rate",
        help="Learning rate for optimizer.",
        type=float,
        default=0.001,
    )
    parser.add_argument(
        "--lite",
        help="If this variable is given, don't store model predictions.",
        action="store_true",
    )
    args = parser.parse_args()

    assert args.data_dir.is_dir()
    assert args.save_dir.is_dir()
    previous_meta = data_dir.joinpath("metadata.json")
    assert previous_meta.is_file()

    # Set device
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Format metadata
    kwargs = {
        "data_dir": str(args.data_dir),
        "split_key": args.split,
        "save_dir": str(args.save_dir),
        "state_key": args.state_key,
        "model_key": args.model,
        "batch_size": args.batch_size,
        "num_epochs": args.num_epochs,
        "learning_rate": args.learning_rate,
        "device": device,
        "lite": args.lite,
    }
    metadata = {"kwargs": kwargs}

    # Create run name
    origin = json.load(previous_meta.open("r"))["name"]
    metadata["name"] = (
        f"split-{args.split}_model-{args.model}_bs-{args.batch_size}_"
        f"epochs-{args.num_epochs}_lr-{args.learning_rate}_{origin}"
    )

    main(metadata=metadata, **kwargs)
