"""
Take in a list of directories with images in them. Grab the brightest pixels,
which we assume to be looking directly at the light. Check how close to white
they are and visualize. We assume that the folder names are R-G-B.
"""

import argparse
import cv2
import matplotlib
from matplotlib import pyplot
import numpy
from pathlib import Path
from tqdm import tqdm


WHITE = numpy.ones(3)
WHITE = WHITE / numpy.linalg.norm(WHITE)


def normalized_light_color(imdir, brightness_percentile=99):
    brightest = numpy.zeros((0, 3), dtype=numpy.uint8)
    for impath in tqdm(sorted(imdir.glob("*.jpg"))):
        rgb = cv2.cvtColor(cv2.imread(str(impath)), cv2.COLOR_BGR2RGB)
        brightness = numpy.linalg.norm(rgb, axis=2)
        cutoff = numpy.percentile(brightness.flatten(), brightness_percentile)
        brightest = numpy.vstack([brightest, rgb[brightness > cutoff]])
    vector = brightest.mean(axis=0)
    normed = vector / numpy.linalg.norm(vector)
    return normed


def scatter_colors(data):
    """Plot colors by R and B (assume G stays constant at 1.0)."""
    figure = pyplot.figure()

    keys = sorted(data.keys())
    x = [data[key]["RGB"][0] for key in keys]
    y = [data[key]["RGB"][2] for key in keys]
    c = [data[key]["color"] for key in keys]
    pyplot.plot([min(x), max(x)], [min(x), max(x)], "r--", label="1:1")
    pyplot.scatter(x, y, c=c, s=1000, edgecolor="k")
    pyplot.xlabel("R")
    pyplot.xlabel("B")

    figure.tight_layout()


def plot_scores(data):
    """Plot how aligned the points are with white"""
    figure = pyplot.figure()

    keys = sorted(data.keys(), key=lambda x: data[x]["white_score"])
    y = [data[k]["white_score"] for k in keys]

    pyplot.bar(keys, y)
    pyplot.xticks(rotation=90)

    yrange = max(y) - min(y)
    pyplot.ylim(min(y) - 0.1 * yrange, max(y) + 0.1 * yrange)

    pyplot.xlabel("Settings")
    pyplot.ylabel("White Alignment (1=perfect)")
    pyplot.title("Color Alignment")

    figure.tight_layout()


def plot_linearity(data):
    """How well controlled was the red/blue value by the settings?"""

    keys = sorted(data.keys())

    def getcolor(i):
        """Get colors, where the measured values are normalized by green"""
        return (
            [data[k]["RGB"][i] for k in keys],
            [data[k]["color"][i] / data[k]["color"][1] for k in keys],
        )

    red = getcolor(0)
    blue = getcolor(2)

    figure, axes = pyplot.subplots(1, 2, figsize=(7, 4.5))

    axes[0].scatter(*red, c="r", s=100, edgecolor="k")
    axes[1].scatter(*blue, c="b", s=100, edgecolor="k")

    axes[0].set_title("Color linearity")
    axes[0].set_xlabel("Red setting")
    axes[0].set_ylabel("Measured Red (green norm)")
    axes[1].set_title("Color linearity")
    axes[1].set_xlabel("Blue setting")
    axes[1].set_ylabel("Measured Blue (green norm)")

    figure.tight_layout()


def main(imdirs):
    data = {
        d.name: {
            "RGB": [float(num) for num in d.name.split("-")],
            "color": normalized_light_color(d),
        }
        for d in imdirs
    }
    for k in data.keys():
        data[k]["white_score"] = data[k]["color"].dot(WHITE)

    for k, v in data.items():
        print(k, v)

    matplotlib.use("TkAgg")
    scatter_colors(data)
    plot_scores(data)
    plot_linearity(data)
    pyplot.show()


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "imdirs",
        help="Space separated paths to image directories, which we assume are"
        " formatted with R-G-B and contain .jpg images.",
        type=Path,
        nargs="+",
    )
    args = parser.parse_args()
    for imdir in args.imdirs:
        assert imdir.is_dir()

    main(args.imdirs)
