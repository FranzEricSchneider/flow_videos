"""
View the trained model on a single video. Note that the training step must
have been run without --lite.
"""

import argparse
import joblib
import json
from matplotlib import pyplot
from matplotlib import animation
import numpy
from pathlib import Path
from PIL import Image
import shutil

from assess.train_classic.allocate import train_test


def extract_state(meta_im, all_y, all_pred, vidfilter):

    # Grab only the images for the video we care about
    all_im = sum(train_test(meta_im), start=[])
    impaths = [im for im in all_im if vidfilter in meta_im[im]["origin"]]
    assert len(impaths) > 0

    # Grab the y (label) and pred (predicted) values as well
    y = numpy.array([all_y[all_im.index(im)] for im in impaths])
    pred = numpy.array([all_pred[all_im.index(im)] for im in impaths])

    # Sort everything according to the original frame order
    ordered = sorted(zip(impaths, y, pred), key=lambda x: meta_im[x[0]]["origin"])

    # The split them back apart, in the same order
    impaths, y, pred = zip(*ordered)

    return impaths, numpy.array(y), numpy.array(pred)


def per_point(axis, y, pred, ylabel):

    axis.plot(y, "k--")
    axis.plot(pred, "o-")

    axis.set_ylabel(ylabel)
    axis.set_xlabel("Frames")
    axis.set_title("Frame by frame comparison to label")


def cumulative(axis, y, ylabel, goal):

    axis.plot([0, len(y)], [goal] * 2, "k--")
    axis.plot(numpy.cumsum(y))

    axis.set_ylabel("Total " + ylabel)
    axis.set_xlabel("Frames")
    axis.set_title("Cumulative state towards the total goal")


def animate_flow(impaths, y, pred, goal, key, save):

    # Make the figure and placeholders for all the lines
    figure, axes = pyplot.subplots(3, 1, figsize=(10, 8))
    (line0,) = axes[0].plot([], [], "o-")
    (line1,) = axes[1].plot([], [])
    image = axes[2].imshow(numpy.zeros_like(numpy.asarray(Image.open(impaths[0]))))
    axes[2].axis("off")

    # Plot the constants
    axes[0].plot(y, "k--")
    axes[1].plot([0, len(y)], [goal] * 2, "k--")

    axes[0].set_ylim((min(pred), max(pred)))
    axes[1].set_ylim((0, sum(pred)))

    # Set up the labels
    axes[0].set_ylabel(key)
    axes[0].set_xlabel("Frames")
    axes[0].set_title("Frame by frame comparison to label")
    axes[1].set_ylabel("Total " + key)
    axes[1].set_xlabel("Frames")
    axes[1].set_title("Cumulative state towards the total goal")
    figure.tight_layout()

    def init():
        line0.set_data([], [])
        line1.set_data([], [])
        image.set_data(numpy.zeros((10, 10)))
        return line0, line1, image

    def update(frame):
        line0.set_data(range(frame), pred[:frame])
        line1.set_data(range(frame), numpy.cumsum(pred[:frame]))
        image.set_data(numpy.asarray(Image.open(impaths[frame])))
        return line0, line1, image

    ani = animation.FuncAnimation(
        figure,
        update,
        frames=len(y),
        init_func=init,
        blit=True,
    )
    ani.save(save / "animation.mp4", writer="ffmpeg", fps=3)


def main(meta1, meta2, trial, vid, key, animate, save):

    # Extract the state label (y) and the state prediction (pred)
    impaths, y, pred = extract_state(
        meta_im=meta1["images"],
        all_y=numpy.hstack([meta2["y_train"], meta2["y_test"]]),
        all_pred=numpy.hstack([meta2["y_pred_train"], meta2["y_pred_test"]]),
        vidfilter=str(Path(trial) / vid),
    )

    # Extract the goal
    goal = meta1["kwargs"]["lookup_dict"][vid][key]

    if animate:
        animate_flow(impaths, y, pred, goal, key, save)

    else:
        figure, axes = pyplot.subplots(2, 1, figsize=(8, 6))
        per_point(
            axis=axes[0],
            y=y,
            pred=pred,
            ylabel=key,
        )
        cumulative(
            axis=axes[1],
            y=pred,
            ylabel=key,
            goal=goal,
        )
        figure.tight_layout()
        pyplot.savefig(save / "inspect.jpg")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--data-dir",
        help="Path to train data with metadata.json.",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--save-dir",
        help="Directory in which to save the graphs.",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--imdir",
        help="Directory in the video trials are stored.",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--trial",
        help="Name of the trial to use.",
        required=True,
    )
    parser.add_argument(
        "--vid",
        help="Name of the video to use.",
        required=True,
    )
    parser.add_argument(
        "--state-key",
        help="Key to look up state we regressed to.",
        required=True,
    )
    parser.add_argument(
        "--animate",
        help="If this is true, make an animation.",
        action="store_true",
    )
    args = parser.parse_args()

    assert args.data_dir.is_dir()
    assert args.imdir.is_dir()

    # Delete the save directory if it exists
    if args.save_dir.is_dir():
        shutil.rmtree(args.save_dir)
    args.save_dir.mkdir()

    # Load the metadata for the two stages (1) preprocess (2) training
    load_meta2 = json.load(args.data_dir.joinpath("metadata.json").open("r"))
    load_meta1 = json.load(
        (Path(load_meta2["kwargs"]["data_dir"]).joinpath("metadata.json").open("r"))
    )
    main(
        meta1=load_meta1,
        meta2=load_meta2,
        trial=args.trial,
        vid=args.vid,
        key=args.state_key,
        animate=args.animate,
        save=args.save_dir,
    )
