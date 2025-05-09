"""
Snip windows out of a set of source images, and record stats about each
snipped subimage. The goal is to take a set of images forming a video and
create a training set of snipped and potentially transformed images. The images
will be sorted into train/test.
"""

import argparse
import json
import numpy
from pathlib import Path
from PIL import Image
from skimage.transform import resize
from sklearn.model_selection import train_test_split
import time
from tqdm import tqdm
from typing import Union


# For now, hard-code the extension we are looking for
EXT = "*.jpg"


# Use this map to divide a Fourier magnitude image into a series of areas. The
# numbers in the map represent the index of the areas
MASK_MAP = numpy.array(
    [
        [6, 7, 8],
        [5, 0, 1],
        [4, 3, 2],
    ]
)
# These will need to be populated later when we have the windows
MASKS = []


def build_masks(window, downsample):
    """
    Populate the global MASKS variables using the available window and
    downsample settings.
    """

    global MASKS
    MASKS = []

    (nw_i, nw_j), (se_i, se_j) = window

    for i in numpy.unique(MASK_MAP):

        # Get the mask for this particular index
        raw_mask = MASK_MAP == i

        # Resize to the window size
        height = (se_i - nw_i) // downsample
        width = (se_j - nw_j) // downsample
        as_float = resize(
            raw_mask.astype(float), (height, width), order=1, anti_aliasing=False
        )

        # Cast as a boolean mask and save
        mask = as_float > 0.5
        MASKS.append({"mask": mask, "count": numpy.sum(mask)})


def get_empty_image(data_dir: Path, lookup_dict: dict) -> numpy.ndarray:
    """
    Get an empty image from the directory that has a None state. Return an
    RGB numpy array.

    Enforce that the minimum empty value is 1 (not 0).
    """

    for imdir, state in lookup_dict.items():
        if any([v is None for v in state.values()]):
            empty_dir = data_dir / imdir
            break
    else:
        raise RuntimeError("Couldn't find an empty dataset")

    # Extract a random empty image
    for impath in empty_dir.glob(EXT):
        image = numpy.asarray(Image.open(impath)).copy()
        image[image == 0] = 1
        return image


def get_impaths(data_dir: Path, lookup_dict: dict, start_stop: dict) -> list:
    """
    Get a list of all pathlib.Paths to images which are not from empty runs
    (None state). Respect the start and stop indices. The paths will be in no
    particular order.

    Assumes that the images have integer names according to their order, e.g.
        1.jpg
        2.jpg ...
    Alternatively,
        0001.jpg
        0002.jpg ...
    """
    impaths = []
    for imdir in data_dir.glob("*/"):

        # Skip empty runs (anything with state of None)
        state = lookup_dict[imdir.name]
        if any([v is None for v in state.values()]):
            continue

        # Find images within the working range
        imrange = start_stop[imdir.name]

        def inrange(impath):
            index = int(impath.stem)
            return (index >= imrange["start"]) & (index <= imrange["stop"])

        impaths.extend(list(filter(inrange, imdir.glob(EXT))))

    return impaths


def imdivide(mat0: numpy.ndarray, mat1: numpy.ndarray) -> numpy.ndarray:
    """
    Divide mat0 by mat1 and re-normalize to a 0-255 image. It is assumed that
    the minimum mat1 value is 1 to avoid divide by 0.
    """
    div = mat0 / mat1
    return (div / div.max() * 255).astype(numpy.uint8)


def imsample(image: numpy.ndarray, downsample: int) -> numpy.ndarray:
    pil = Image.fromarray(image)
    size = (pil.width // downsample, pil.height // downsample)
    return numpy.asarray(pil.resize(size, resample=Image.LANCZOS))


def snip_report(
    metadata: dict,
    impath: Path,
    empty: Union[None, numpy.ndarray],
    windows: list,
    downsample: int,
    save_dir: Path,
    index: int,
) -> None:

    # Load the base image
    image = Image.open(impath)

    for i, ((nw_i, nw_j), (se_i, se_j)) in enumerate(windows):

        # Make the initial window crop
        snippet = numpy.asarray(image.crop((nw_j, nw_i, se_j, se_i)))

        # Divide by empty (if relevant) and re-norm to 0-255 + uint8
        if empty is not None:
            snippet = imdivide(snippet, empty[nw_i:se_i, nw_j:se_j])

        # Downsample (if relevant)
        if downsample > 1:
            snippet = imsample(snippet, downsample)

        # Save the image
        outpath = save_dir / f"{index:05}-{i}.jpg"
        Image.fromarray(snippet).save(outpath)

        # Record any stats
        metadata["images"][str(outpath)] = {
            "origin": str(impath),
            "window": windows[i],
            "window_index": i,
            "stats": calc_stats(snippet),
        }


def calc_stats(rgb: numpy.ndarray) -> dict:
    """
    Calculate a dictionary of various stats. Some will be just on a single
    type of image:
        Fourier noise spectrum
        Histogram
    And some will be on a variety of images (grayscale, RGB, HSV)
        Average
        Std deviation
        Shannon entropy
    The stats should strain to be *relatively* low dimension
    """

    # Calculate our alternatives to RGB
    pil = Image.fromarray(rgb)
    gray = numpy.asarray(pil.convert("L"))
    hsv = numpy.asarray(pil.convert("HSV"))

    stats = {}

    # Spectral data
    mag, phase = fourier_vector(gray)
    stats["fourier-magnitude"] = mag
    stats["fourier-phase"] = phase

    # Brightness histogram
    stats["gray-histogram"] = histogram(gray)

    for imtype, image in [("gray", gray), ("rgb", rgb), ("hsv", hsv)]:
        stats[f"{imtype}-average"] = average(image)
        stats[f"{imtype}-std"] = stddev(image)
        stats[f"{imtype}-entropy"] = entropy(image)
        stats[f"{imtype}-histogram"] = histogram(image)

    # Convert to JSON possible values
    for key, value in stats.items():
        if isinstance(value, numpy.ndarray):
            stats[key] = value.tolist()

    return stats


def fourier_vector(gray: numpy.ndarray) -> tuple:
    """
    Calculate vectors of the magnitude and phase according to the MASKS layout.
    """

    # Also shift the Fourier data so it is centered
    fourier = numpy.fft.fftshift(numpy.fft.fft2(gray))

    # Calculate a log magnitude (more human readable)
    magnitude = numpy.log(1 + numpy.abs(fourier))

    # And the phase
    phase = numpy.angle(fourier)

    # Create a normalized zone vector
    def vectorize(array):
        return numpy.array(
            [numpy.sum(array[mdata["mask"]]) / mdata["count"] for mdata in MASKS]
        )

    return vectorize(magnitude), vectorize(phase)


def average(image: numpy.ndarray) -> Union[float, numpy.ndarray]:
    """Take the average based on 2D or 3D."""
    if len(image.shape) == 2:
        return float(image.mean())
    elif len(image.shape) == 3:
        return image.mean(axis=(0, 1))
    else:
        raise ValueError(f"Unexpected shape: {image.shape}")


def stddev(image: numpy.ndarray) -> Union[float, numpy.ndarray]:
    """Take the std dev based on 2D or 3D."""
    if len(image.shape) == 2:
        return float(image.std())
    elif len(image.shape) == 3:
        return image.std(axis=(0, 1))
    else:
        raise ValueError(f"Unexpected shape: {image.shape}")


def entropy(image: numpy.ndarray) -> Union[float, numpy.ndarray]:
    """Take the entropy based on 2D or 3D."""
    if len(image.shape) == 2:
        return float(shannon_entropy_1d(image))
    elif len(image.shape) == 3:
        return numpy.array(
            [shannon_entropy_1d(image[:, :, i]) for i in range(image.shape[2])]
        )
    else:
        raise ValueError(f"Unexpected shape: {image.shape}")


def shannon_entropy_1d(gray: numpy.ndarray) -> float:
    # Calculate the normalized weight of each pixel value, 0-255
    hist, _ = numpy.histogram(gray, bins=256, range=(0, 256), density=True)
    # Avoid log(0)
    hist = hist[hist > 0]
    # Use the base-2 logarithm to run this Shannon entropy calculation
    return -numpy.sum(hist * numpy.log2(hist))


def histogram(gray: numpy.ndarray, N=10) -> numpy.ndarray:
    # Calculate the normalized weight of binned pixel values
    hist, _ = numpy.histogram(gray, bins=N, range=(0, 256), density=True)
    return hist


def main(
    metadata: dict,
    data_dir: str,
    start_stop: dict,
    lookup_dict: dict,
    windows: list,
    downsample: int,
    train_frac: float,
    divide_empty: bool,
    save_dir: str,
):

    # Convert back to pathlib
    data_dir = Path(data_dir)

    # Populate the Fourier masks with an arbitrary window
    build_masks(windows[0], downsample)

    # Create a place to save everything
    timestamp = int(time.time() * 1e6)
    out_dir = Path(save_dir) / f"images_{timestamp}"
    train_dir = out_dir / "train"
    test_dir = out_dir / "test"
    for imdir in [out_dir, train_dir, test_dir]:
        imdir.mkdir()

    # If we are dividing by empty, load an empty image
    empty = None
    if divide_empty:
        empty = get_empty_image(data_dir, lookup_dict)

    # Get all non-empty image paths
    impaths = get_impaths(data_dir, lookup_dict, start_stop)

    # Split into groups
    train, test = train_test_split(impaths, train_size=train_frac, random_state=41)

    # Snip windows from each image and save data on them in metadata
    metadata["images"] = {}
    for save, impaths in [(train_dir, train), (test_dir, test)]:
        for i, impath in enumerate(tqdm(impaths, desc=save.name)):
            snip_report(metadata, impath, empty, windows, downsample, save, i)

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
        help="Directory in which all folders represent a video.",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--start-stop",
        help="Each folder in data_dir should have {folder: {start: idx,"
        " stop: idx}} defined in this json file. The start and stop indices"
        " are hand gathered, and delineate where flow starts happening.",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--lookup-dict",
        help="Each folder in data_dir should have {folder: {relevant state}}"
        " defined in this json file. This is the alternative to parsing state"
        " from the file names, which is bad form.",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--windows",
        help="This json should define the windows we want to clip, in the form"
        " [ [(NW i, NW j), (SE i, SE j)], ... ]. There can be multiple. We"
        " assume all windows are the same size.",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--downsample",
        help="How much to downsample clipped images by.",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--train-frac",
        help="Fraction of images that should be designated as training set.",
        type=float,
        default=0.7,
    )
    parser.add_argument(
        "--divide-empty",
        help="If flag is given, divide the images by the first empty image"
        " (determined for now by a video with a state of None).",
        action="store_true",
    )
    parser.add_argument(
        "--save-dir",
        help="Directory in which to save the output clipped images.",
        type=Path,
        required=True,
    )
    args = parser.parse_args()

    assert args.data_dir.is_dir()
    assert args.start_stop.is_file()
    assert args.lookup_dict.is_file()
    assert args.windows.is_file()
    assert args.save_dir.is_dir()
    assert args.downsample >= 1
    assert 0.0 < args.train_frac < 1.0

    # Load the settings files
    start_stop = json.load(args.start_stop.open("r"))
    lookup_dict = json.load(args.lookup_dict.open("r"))
    windows = json.load(args.windows.open("r"))

    # Format this in a saveable fashion
    kwargs = {
        "data_dir": str(args.data_dir),
        "data_name": args.data_dir.name,
        "start_stop": start_stop,
        "lookup_dict": lookup_dict,
        "window_file": args.windows.stem,
        "windows": windows,
        "downsample": args.downsample,
        "train_frac": args.train_frac,
        "divide_empty": args.divide_empty,
        "save_dir": str(args.save_dir),
    }
    metadata = {"kwargs": kwargs}

    # Create a name for this run
    metadata["name"] = (
        f"dir-{args.data_dir.name}_{args.windows.stem}_ds-{args.downsample}_norm-{args.divide_empty}"
    )

    # And call
    main(metadata=metadata, **kwargs)


# TODO: Make a metadata ingester
