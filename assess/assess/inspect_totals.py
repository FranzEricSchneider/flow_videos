"""Script to sum inferences per video and compare to lookup_dict state for each video.
Handles both train_classic and train_cnn result formats.
"""

import argparse
from collections import defaultdict
import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path


def sum_predictions(meta1, meta2, state_key):

    # Get all image paths and predictions
    all_paths = meta2["train_image_paths"] + meta2["test_image_paths"]
    all_pred = meta2["y_pred_train"] + meta2["y_pred_test"]

    # Map from video name to (sum of predictions, state)
    video_pred_sum = defaultdict(float)
    video_state = {}
    for i, im in enumerate(all_paths):

        # Get video key and name from origin (assume origin is path/to/video/frame...)
        origin = Path(meta1["images"][im]["origin"])
        video_key = str(origin.parent)
        video = origin.parent.name

        # Sum predictions for this video
        video_pred_sum[video_key] += all_pred[i]

        # Get state from lookup_dict
        if video_key not in video_state:
            video_state[video_key] = meta1["kwargs"]["lookup_dict"][video][state_key]

    # Prepare output lists
    state_list = []
    pred_sum_list = []
    for video_key, state in video_state.items():
        state_list.append(state)
        pred_sum_list.append(video_pred_sum[video_key])
    return state_list, pred_sum_list


def main():
    parser = argparse.ArgumentParser(
        description="Sum inferences per video and compare to lookup_dict state."
    )
    parser.add_argument(
        "--result-dirs",
        nargs="+",
        type=Path,
        required=True,
        help="List of result directories (from train_classic or train_cnn)",
    )
    parser.add_argument(
        "--names",
        nargs="+",
        type=str,
        required=True,
        help="Human-readable names for each result dir (must match length)",
    )
    parser.add_argument(
        "--state-key", type=str, required=True, help="State key to use from lookup_dict"
    )
    parser.add_argument(
        "--save-dir", type=Path, required=True, help="Directory to save plot.png"
    )
    args = parser.parse_args()

    if len(args.result_dirs) != len(args.names):
        raise ValueError("--names must be the same length as --result-dirs")

    args.save_dir.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(7, 5))
    max_state = 0
    for result_dir, name in zip(args.result_dirs, args.names):
        meta2 = json.load((result_dir / "metadata.json").open("r"))
        meta1 = json.load(
            (Path(meta2["kwargs"]["data_dir"]) / "metadata.json").open("r")
        )
        state_list, pred_sum_list = sum_predictions(meta1, meta2, args.state_key)
        error = [
            float(pred) - float(state) for pred, state in zip(pred_sum_list, state_list)
        ]
        rmse = np.sqrt(np.mean(np.square(error)))
        plt.scatter(
            state_list, pred_sum_list, label=f"{name}, RMSE={rmse:.1f}", marker="x"
        )
        if max(state_list) > max_state:
            max_state = max(state_list)

    plt.plot(
        [0, max_state],
        [0, max_state],
        color="black",
        linestyle="--",
        linewidth=1,
        label="Ideal",
    )
    plt.xlabel(f"Total (ground truth) per video [{args.state_key}]")
    plt.ylabel(f"Summed inference [{args.state_key}]")
    plt.title("Summed inference vs. ground truth total, by model and video")
    plt.legend()
    plt.tight_layout()
    plt.savefig(args.save_dir / "plot.png")
    plt.close()


if __name__ == "__main__":
    main()
