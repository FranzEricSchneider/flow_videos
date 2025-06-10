"""
Tools to take a state measurement per video, and allocate it across the
frames in that video.

For example, a video may have 180 active frames and 90 grams of material
passed by. We can assume that it was split evenly (0.5 g / frame) or we could
make more complex assumptions about how the flow was distributed.
"""

import numpy
from pathlib import Path


def get_state(impath: Path, lookup_dict: dict, start_stop: dict, key: str) -> dict:
    """
    Given an image path, look up the state and  first and last frames that
    were active, then return those alongside the image path index in a dict.
    """
    impath = Path(impath)
    parent = impath.parent.name
    return {
        "state": lookup_dict[parent][key],
        "first": start_stop[parent]["start"],
        "last": start_stop[parent]["stop"],
        "index": int(impath.stem),
    }


def split_evenly(state: float, first: int, last: int, index: int) -> float:
    """
    In this particular function, we return the same no matter the index, it is
    simply the state evenly divided across all the frames.
    """
    assert first <= index <= last, f"Bad order: {first}<={index}<={last}"
    delta = last - first
    return state / delta


def split_ramp(state: float, first: int, last: int, index: int) -> float:
    """Ramp up to the midpoint, then ramp down."""
    assert first <= index <= last, f"Bad order: {first}<={index}<={last}"
    raise NotImplementedError()


def split_trapezoid(
    state: float, first: int, last: int, index: int, max_frac: float
) -> float:
    """
    Ramp up to the midpoint, then ramp down. The max fraction is the fraction
    (0 - 0.5) of the video at which we reach our maximum value.
      ______
     /      \
    /        \
    """
    assert first <= index <= last, f"Bad order: {first}<={index}<={last}"
    assert 0 < max_frac < 0.5, f"Max fraction ({max_frac}) should be 0 - 0.5"
    raise NotImplementedError()


def split_brightness(
    state: float, first: int, last: int, index: int, brightness: numpy.ndarray
) -> float:
    """
    Split according to the average by-image brightness (darker = higher
    proportion of the state).
    """
    assert first <= index <= last, f"Bad order: {first}<={index}<={last}"
    assert len(brightness) == (last - first), "Brightness not aligned to frames"
    raise NotImplementedError()


def split_model(state: float, first: int, last: int, index: int, model) -> float:
    """Use an already trained model to divide the state amongst frames."""
    assert first <= index <= last, f"Bad order: {first}<={index}<={last}"
    raise NotImplementedError()


ALLOCATE = {
    "even": split_evenly,
    "ramp": split_ramp,
    "trap": split_trapezoid,
    "grey": split_brightness,
    "model": split_model,
}


def train_test(imdict):

    train = []
    test = []

    for impath in imdict.keys():
        if Path(impath).parent.name == "train":
            train.append(impath)
        elif Path(impath).parent.name == "test":
            test.append(impath)
        else:
            raise ValueError(f"Unknown path {impath}")

    return sorted(train), sorted(test)
