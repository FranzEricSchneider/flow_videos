"""
Take in a list of images. Assume that each image has a mask.png of the same
size in the same folder, load a mask from the red portion of the mask image,
then assess the brightness under that mask.
"""

import argparse
from collections import defaultdict
import cv2
import matplotlib
from matplotlib import pyplot
import numpy
from pathlib import Path
from tqdm import tqdm
from scipy.optimize import curve_fit


def get_mask(impath, color):
    rgb = cv2.cvtColor(cv2.imread(str(impath)), cv2.COLOR_BGR2RGB)
    return numpy.all(rgb == color, axis=2)


def get_pixels(impath, mask):
    """Get grayscale pixel values under the mask as a vector."""
    gray = cv2.cvtColor(cv2.imread(str(impath)), cv2.COLOR_BGR2GRAY)
    return gray[mask]


def view_hist(data, xlabel):
    """Plot each image's data as a histogram"""

    # Set up color mapping
    colors = pyplot.cm.tab10(numpy.linspace(0, 1, len(data)))

    # Plot each group of vectors
    figure = pyplot.figure(figsize=(10, 6))
    for i, (label, vectors) in enumerate(data.items()):
        for vector in vectors:
            pyplot.hist(
                vector,
                bins=30,
                density=True,
                alpha=0.2,
                edgecolor="black",
                color=colors[i],
                label=label if vector is vectors[0] else None,
            )

    pyplot.legend(title="Group")
    pyplot.xlabel(xlabel)
    pyplot.ylabel("Normalized Brightness")
    pyplot.title("Normalized Brightness Histograms by Group")
    figure.tight_layout()


def view_exponential(data, x, y, params, xlabel):
    """See the brightness data as an exponential"""

    figure = pyplot.figure(figsize=(6, 4))

    pyplot.scatter(x, y, c="b", s=100, edgecolor="k", label="datapoints")

    fit_x = numpy.linspace(min(x), max(x))
    fit_y = decay(fit_x, *params)
    pyplot.plot(fit_x, fit_y, "g", label="fit")

    a, b, c = params
    pyplot.title(
        "Exponential fit: $" + f"{a:.1f}" + "e^{" + f"-{b:.1f}" + "x}$ + " + f"{c:.1f}"
    )
    pyplot.legend()
    pyplot.xlabel(xlabel)
    pyplot.ylabel("Average Brightness")
    figure.tight_layout()


def decay(times, a, b, c):
    return a * numpy.exp(-b * times) + c


def calc_exponential(data, initial=(500, 1, 25)):
    """
    Calculate the exponential form for brightness data. Note that the initial
    guess is fairly necessary for convergence.
    """
    x = []
    y = []
    for xval, vectors in data.items():
        for vector in vectors:
            x.append(xval)
            y.append(vector.mean())
    params, covariance = curve_fit(decay, x, y, p0=initial)
    return x, y, params


def main(impaths, xvals, xlabel):
    data = defaultdict(list)
    for p, x in zip(tqdm(impaths), xvals):
        mask = get_mask(p.parent / "mask.png", [255, 0, 0])
        data[x].append(get_pixels(p, mask))

    for k, v in data.items():
        print(f"X: {k}")
        for pixels in v:
            print(f"Mean: {pixels.mean():.1f}, std: {pixels.std():.1f}")

    x, y, params = calc_exponential(data)

    matplotlib.use("TkAgg")
    view_hist(data, xlabel)
    view_exponential(data, x, y, params, xlabel)
    pyplot.show()


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "-i",
        "--impaths",
        help="Space separated paths to images, which we assume have matching"
        " mask.png images in the same folder.",
        type=Path,
        nargs="+",
        required=True,
    )
    parser.add_argument(
        "-x",
        "--x-values",
        help="Space separated X values to match with the paths.",
        type=int,
        nargs="+",
        required=True,
    )
    parser.add_argument(
        "-X", "--x-label", help="Plot label for the X axis.", default="X Axis"
    )
    args = parser.parse_args()
    for imdir in args.impaths:
        assert imdir.is_file()
    assert len(args.x_values) == len(args.impaths)

    main(args.impaths, args.x_values, args.x_label)
