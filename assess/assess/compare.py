"""Collate and compare results across training runs."""

import argparse
from functools import reduce
import joblib
import json
from matplotlib import pyplot
import operator
from pathlib import Path
from sklearn.metrics import mean_absolute_error, mean_squared_error
import time
from tqdm import tqdm


def plot_ranking(metadatas: list, origins: dict, stats: list, N: int = 30):

    highest_best = {
        "R2": True,
        "RMSE": False,
        "MAE": False,
    }

    order = sorted(
        range(len(metadatas)),
        key=lambda x: metadatas[x][f"{stats[0]}_test"],
        reverse=highest_best[stats[0]],
    )

    # Take a subset of the tests - just the top and bottom slices
    if len(order) <= 2 * N:
        subset = order
    else:
        subset = order[:N] + order[-N:]

    for key in ["train", "test"]:

        figure, axes = pyplot.subplots(len(stats), 1, figsize=(10, 2 + 3 * len(stats)))
        if len(stats) == 1:
            axes = [axes]

        figure.suptitle(f"Sorted stats for {key} set")

        for axis, stat in zip(axes, stats):
            axis.set_title(f"{stat}, sorted")
            axis.set_ylabel(stat)

            x = [
                f"{origins[metadatas[i]['kwargs']['data_dir']]['name']}\n{metadatas[i]['name']}"
                for i in subset
            ]
            y = [metadatas[i][f"{stat}_{key}"] for i in subset]

            axis.bar(x, y)
            if stat == stats[-1]:
                axis.tick_params(axis="x", labelrotation=90, labelsize=6)
            else:
                axis.tick_params(
                    axis="x", which="both", bottom=False, labelbottom=False
                )

        figure.tight_layout()

    pyplot.show()


def plot_vs(metadatas: list, origins: dict, stats: str, variable: str):

    for key in ["train", "test"]:

        for stat in stats:

            figure, axis = pyplot.subplots(1, 1, figsize=(4.5, 2.5))
            figure.suptitle(f"{stat} for {key} set, sorted by {variable}")
            axis.set_ylabel(stat)

            x = [lookup(meta, origins, variable) for meta in metadatas]
            y = [meta[f"{stat}_{key}"] for meta in metadatas]

            axis.scatter(x, y)
            axis.grid(True)
            axis.tick_params(axis="x", labelrotation=90, labelsize=8)

            figure.tight_layout()

    pyplot.show()


def lookup(meta, origins, variable):
    """
    According to some hand-coded logic, parse a period-separated string
    specifying a variable to its value
    """
    keys = variable.split(".")
    source = keys.pop(0)

    if source == "meta":
        data = meta
    elif source == "origin":
        data = origins[meta["kwargs"]["data_dir"]]
    else:
        raise ValueError(f"Source {source} not an expected value")

    return reduce(operator.getitem, keys, data)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--data-dir",
        help="Path to folder with training folders inside.",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--stats",
        help="Space separated results to view.",
        required=True,
        choices=["R2", "RMSE", "MAE"],
        nargs="+",
    )
    parser.add_argument(
        "--plot-ranking",
        help="Whether to plot a ranking of all runs.",
        action="store_true",
    )
    parser.add_argument(
        "--plot-vs-variable",
        help="Scatter performance against a single variable. See the details"
        " of lookup() for how specific variables are parsed from metadata.",
    )
    parser.add_argument(
        "--exclude",
        help="Space separate values of variable:value. Any runs where the"
        " variable is equal to that value will be expcluded. See the details"
        " of lookup() for how specific variables are parsed from metadata.",
        nargs="+",
    )
    args = parser.parse_args()

    assert args.data_dir.is_dir()

    # Track down all available metadata files
    metadatas = [
        json.load(path.open("r"))
        for path in tqdm(
            sorted(args.data_dir.glob("*/metadata.json")),
            desc="Loading metadata",
        )
    ]
    origins = {}
    for meta in tqdm(metadatas, desc="Matching origins"):
        key = meta["kwargs"]["data_dir"]
        if key not in origins:
            origins[key] = json.load((Path(key).joinpath("metadata.json").open("r")))

    # Keep only metadata that does not have excluded data
    if args.exclude is None:
        remaining = metadatas
    else:
        remaining = []
        for meta in metadatas:
            for exclude in args.exclude:
                variable, value = exclude.split(":")
                metavalue = lookup(meta, origins, variable)
                if metavalue == type(metavalue)(value):
                    break
            else:
                remaining.append(meta)

    if args.plot_ranking:
        plot_ranking(remaining, origins, args.stats)

    if args.plot_vs_variable is not None:
        plot_vs(remaining, origins, args.stats, args.plot_vs_variable)
