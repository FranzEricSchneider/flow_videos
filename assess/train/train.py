"""Tool to train various kinds of regressor on image snippets."""

import argparse
import joblib
import json
import numpy
from pathlib import Path
from sklearn.metrics import mean_absolute_error, mean_squared_error
import time

from allocate import ALLOCATE, get_state, train_test
from model import get_model, MODELS


def main(
    metadata: dict,
    data_dir: str,
    split_key: str,
    save_dir: str,
    stat: str,
    state_key: str,
    model_key: str,
    scale: bool,
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
    train, _ = train_test(previous["images"])

    # Build the vector set X of shape (N, M)
    X = numpy.vstack([previous["images"][im]["stats"][stat] for im in train])
    metadata["X"] = X.tolist()

    # Also build the label set y of shape (N,)
    y = numpy.hstack(
        [
            splitter(
                **get_state(
                    previous["images"][im]["origin"],
                    previous["kwargs"]["lookup_dict"],
                    previous["kwargs"]["start_stop"],
                    state_key,
                )
            )
            for im in train
        ]
    )
    metadata["y"] = y.tolist()

    # Train and save the model
    regressor = get_model(key=model_key, scale=scale)
    regressor.fit(X, y)
    joblib.dump(regressor, out_dir / "model.joblib")

    # Save a few training stats
    metadata["R2"] = regressor.score(X, y)
    metadata["RMSE"] = numpy.sqrt(mean_squared_error(y, regressor.predict(X)))
    metadata["MAE"] = mean_absolute_error(y, regressor.predict(X))

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
    args = parser.parse_args()

    assert args.data_dir.is_dir()

    # Format this in a saveable fashion
    kwargs = {
        "data_dir": str(args.data_dir),
        "split_key": args.split,
        "save_dir": str(args.save_dir),
        "stat": str(args.stat),
        "state_key": str(args.state_key),
        "model_key": str(args.model),
        "scale": str(args.scale),
    }
    metadata = {"kwargs": kwargs}

    # Create a name for this run
    metadata["name"] = (
        f"split-{args.split}_stat-{args.stat}_model-{args.model}_scale-{args.scale}"
    )

    main(metadata=metadata, **kwargs)


# TODO: Make a metadata ingester
