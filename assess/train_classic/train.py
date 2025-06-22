"""Tool to train various kinds of regressor on image snippets."""

import argparse
import joblib
import json
import numpy
from pathlib import Path
from sklearn.metrics import mean_absolute_error, mean_squared_error
import time

from assess.train_classic.allocate import ALLOCATE, get_state, train_test
from assess.train_classic.model import get_model, MODELS


def get_vector(imdict: dict, stat: str, imlist: list) -> numpy.ndarray:
    """Build the vector set X of shape (N, M)."""
    return numpy.vstack([imdict[im]["stats"][stat] for im in imlist])


def get_labels(splitter, metadata: dict, key: str, imlist: list) -> numpy.ndarray:
    """Build the label set y of shape (N,)."""
    return numpy.hstack(
        [
            splitter(
                **get_state(
                    metadata["images"][im]["origin"],
                    metadata["kwargs"]["lookup_dict"],
                    metadata["kwargs"]["start_stop"],
                    key,
                )
            )
            for im in imlist
        ]
    )


def main(
    metadata: dict,
    data_dir: str,
    split_key: str,
    save_dir: str,
    stat: str,
    state_key: str,
    model_key: str,
    scale: bool,
    lite: bool,
):

    # Convert back to pathlib
    data_dir = Path(data_dir)

    # Create a place to save everything
    timestamp = int(time.time() * 1e6)
    out_dir = Path(save_dir) / f"{data_dir.name}_{timestamp}"
    out_dir.mkdir()

    # Load the previous step (in this case data prep)
    previous = json.load(data_dir.joinpath("metadata.json").open("r"))

    # Choose a method for allocating state to individual frames
    splitter = ALLOCATE[split_key]

    # Load train vs. test keys in the dictionary
    train, test = train_test(previous["images"])

    # Build the vector set X of shape (N, M)
    X = get_vector(previous["images"], stat, train)

    # Also build the label set y of shape (N,)
    y = get_labels(splitter, previous, state_key, train)

    # Train and save the model
    regressor = get_model(key=model_key, scale=scale)
    regressor.fit(X, y)
    joblib.dump(regressor, out_dir / "model.joblib")

    # Record the train and test set
    y_pred_train = regressor.predict(X)
    X_test = get_vector(previous["images"], stat, test)
    y_test = get_labels(splitter, previous, state_key, test)
    y_pred_test = regressor.predict(X_test)
    if not lite:
        metadata["X_train"] = X.tolist()
        metadata["y_train"] = y.tolist()
        metadata["y_pred_train"] = y_pred_train.tolist()
        metadata["X_test"] = X_test.tolist()
        metadata["y_test"] = y_test.tolist()
        metadata["y_pred_test"] = y_pred_test.tolist()
        metadata["train_image_paths"] = train
        metadata["test_image_paths"] = test

    # Save a few stats
    for kX, ky, kyp, key in [
        (X, y, y_pred_train, "train"),
        (X_test, y_test, y_pred_test, "test"),
    ]:
        metadata[f"R2_{key}"] = regressor.score(kX, ky)
        metadata[f"RMSE_{key}"] = numpy.sqrt(mean_squared_error(ky, kyp))
        metadata[f"MAE_{key}"] = mean_absolute_error(ky, kyp)

    # Finally, save the metadata
    json.dump(
        metadata,
        out_dir.joinpath("metadata.json").open("w"),
        indent=2,
        sort_keys=True,
    )


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
        choices=sorted(ALLOCATE.keys()),
        default="even",
    )
    parser.add_argument(
        "--stat",
        help="Options for which stat to regress against state.",
        choices=[
            "fourier-magnitude",
            "fourier-phase",
            "gray-average",
            "gray-entropy",
            "gray-histogram",
            "gray-std",
            "hsv-average",
            "hsv-entropy",
            "hsv-histogram",
            "hsv-std",
            "rgb-average",
            "rgb-entropy",
            "rgb-histogram",
            "rgb-std",
        ],
        default="gray-average",
    )
    parser.add_argument(
        "--model",
        help="Options for basic regressors.",
        choices=sorted(MODELS.keys()),
        default="ridge",
    )
    parser.add_argument(
        "--scale",
        help="Whether to preprocess rescale the input vectors (good practice).",
        action="store_true",
    )
    parser.add_argument(
        "--lite",
        help="If this variable is given, don't store model or high-res results.",
        action="store_true",
    )
    args = parser.parse_args()

    assert args.data_dir.is_dir()

    # Format this in a saveable fashion
    kwargs = {
        "data_dir": str(args.data_dir),
        "split_key": args.split,
        "save_dir": str(args.save_dir),
        "stat": args.stat,
        "state_key": args.state_key,
        "model_key": args.model,
        "scale": args.scale,
        "lite": args.lite,
    }
    metadata = {"kwargs": kwargs}

    # Create a name for this run
    metadata["name"] = (
        f"split-{args.split}_stat-{args.stat}_model-{args.model}_scale-{args.scale}"
    )

    main(metadata=metadata, **kwargs)


# TODO: Make a metadata ingester
