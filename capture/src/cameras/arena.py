"""
Connect to the Lucid cameras and manage the images they take.
"""

import argparse
import cv2
import json
import logging
import numpy
from pathlib import Path
import time
from tqdm import tqdm

from arena_api.enums import PixelFormat
from arena_api.system import system
from arena_api.buffer import BufferFactory


class ArenaCam:

    # Parameter values that should be checked for approximate equality, not
    # exact settings (depends on camera characteristics)
    KNOWN_APPROXIMATES = [
        "BalanceRatio",
        "ExposureTime",
        "ExposureAutoLowerLimit",
        "ExposureAutoUpperLimit",
        "Gain",
    ]

    def __init__(self, config, log):
        """
        Saves a few class variables and connects to the given camera

        Arguments:
            config: dictionary with all our desired camera settings
            log: logging object

        Returns: Nothing
        """
        self.config = config
        self.log = log
        self.__connect(self.config)

    def __connect(self, config):
        """
        Connects to the chosen camera, sets desired config values. The broad
        strokes are:
            1) Call __connect_device() to start communication
            2) Set a variety of simple settings from config
            3) Set the more complicated settings, such as metadata, AOI, and
                pixel format
            4) Start the camera stream

        Arguments: See __init__()

        Returns: Nothing
        """

        # Connect to the camera based on the specified IP or serial number
        self.serial = config.get("serial", None)
        self.desired_ip = config["ip"]
        self.device, self.device_info = self.__connect_device()
        self.ip = self.device_info["ip"]

        # Set a variety of parameters to the camera
        self.stream_params = self.device.tl_stream_nodemap
        self.camera_params = self.device.nodemap
        for options, param_map in (
            (config["stream"], self.stream_params),
            (config["camera"], self.camera_params),
        ):
            for key, value in options.items():
                self.set_param(key, value, param_map)
        # Set camera to return the specified metadata values in "image chunks"
        for metadata in config["metadata"]:
            self.set_param("ChunkSelector", metadata, self.camera_params)
            self.set_param("ChunkEnable", True, self.camera_params)

        # Configure the areas of interest (AOIs)
        if config.get("AutoExposureAOIEnable", False) is True:
            self.set_AOI("AutoExposureAOI", config["AOI"])
        if config["camtype"] == "color" and config.get("AwbAOIEnable", False) is True:
            self.set_AOI("AwbAOI", config["AOI"])

        if config["camtype"] == "color" and config.get("RGB", None) is not None:
            self.set_white_balance(config["RGB"])

        if config["format"] == "BGR":
            self.pixel_format = PixelFormat.BGR8
        # TODO: Debug. Using Mono causes the color conversion to break down
        # in the Bayer combination step
        # elif config["format"] == "Mono":
        #     self.pixel_format = PixelFormat.Mono8
        else:
            raise ValueError(f"Unknown pixel format {config['format']}")

        # Start the camera streaming after all params are set
        self.device.start_stream()

    def __connect_device(self, tries_max=6, sleep_s=10):
        """
        Waits for the user to connect a device, raising an error on failure if
        we've tried too many times. Tries to connect to a given serial number
        or IP address.

        Arguments:
            tries_max: Number of times to try to connect to the given serial
                number or IP address
            sleep_s: Number of seconds to wait between attempts

        Returns: On success, returns tuple
            [0] Arena camera object
            [1] Dictionary of device info for this camera, returned by
                arena_api.system.device_infos. This is what we check the IP or
                serial against
        """
        device = None
        for tries in range(tries_max):

            device_infos = system.device_infos

            if self.serial is not None:
                key = "serial"
                test = str(self.serial)
            else:
                key = "ip"
                test = self.desired_ip
            info = [di for di in device_infos if di[key] == test]

            if len(info) > 0:
                assert (
                    len(info) == 1
                ), f"Found {len(info)} devices with the same {key}?\n\t{info}"
                self.log.info(f"Connecting to camera: {info}")
                device = system.create_device(info[0])[0]
                break
            else:
                self.log.info(
                    f"Try {tries + 1} of {tries_max}: waiting for {sleep_s}"
                    f" secs for {key}: {test} to be connected! Available"
                    f" devices: {device_infos}"
                )
                for count in range(sleep_s):
                    time.sleep(1)
                    self.log.info(f"{count + 1} seconds passed {'.' * count}")
                tries += 1

        else:
            raise RuntimeError("Specified camera not found! Connect device")

        return device, info[0]

    def set_param(self, name, value, param_map):
        """
        Set a camera parameter to a given param map. Will fail if the value is
        not set correctly (checks after setting).

        Arguments:
            name: Name of the value we want to set (possibilities and effects
                determined by Lucid)
            value: Value we want to set the camera variable to
            param_map: Dictionary-esque access into the camera settings, can
                be used to get or set values

        Returns: Nothing (raises an exception on failure)
        """

        try:
            param_map[name].value = value
        except Exception as connection_error:
            # I know it's bad form to catch "Exception", unfortunately the
            # camera system raises raw Exception values in several
            # circumstances. Adding additional information here for debugging
            self.log.info(
                f"Tried setting {name} to {value}."
                f" Current settings: {param_map[name]}"
            )
            if "SC_ERR_ACCESS_DENIED -1005" in str(connection_error):
                raise RuntimeError(
                    f"Error {str(connection_error)} raised, is there a chance"
                    " another process is already accessing this device?"
                )
            else:
                raise

        if name in ArenaCam.KNOWN_APPROXIMATES:
            # For some floating point values, check that we are within X%
            # of the set value (rtol = relative tolerance)
            success = numpy.isclose(value, param_map[name].value, rtol=0.02)
        else:
            success = param_map[name].value == value
        statement = f"setting {name} to {value}"
        self.log.info(
            f"{self.ip}:\t{statement:50} {'Success' if success else 'FAILURE'}"
        )
        if not success:
            print(param_map[name])
            raise RuntimeError(
                f"After setting {name} found inappropriate value {value}!"
                " (Should this check be approximate?)"
            )

    def set_white_balance(self, rgb):
        for name, value in zip(["Red", "Green", "Blue"], rgb):
            self.set_param("BalanceRatioSelector", name, self.camera_params)
            self.set_param("BalanceRatio", float(value), self.camera_params)

    def set_AOI(self, prefix, AOI, default=(0, 0, 4024, 3036)):
        """
        Set camera parameters for some area of interest.

        Arguments:
            prefix: (string) Lucid follows a convention for various AOI values,
                a prefix and then a fixed set of named values
            AOI: (list/tuple) Four values that we want to set the AOI to
                (Width, Height, OffsetX, OffsetY). The offsets are the initial
                corner of the AOI
            default: What we want to reset the values to initially (should be
                fixed unless we get a different camera with a different image
                size)

        Returns: Nothing
        """

        # First we have to flush the old in case they limit allowable values.
        # NOTE: OffsetX/Y and Width/Height are flipped for permission reasons
        for name, value in zip(["OffsetX", "OffsetY", "Width", "Height"], default):
            self.set_param(f"{prefix}{name}", value, self.camera_params)

        # Then set the desired values
        for name, value in zip(["Width", "Height", "OffsetX", "OffsetY"], AOI):
            self.set_param(f"{prefix}{name}", value, self.camera_params)

    def grab_vidframe(self):
        """Grab an image off the buffer, and metadata if available."""

        # Grab the image from the camera
        buffer = self.device.get_buffer()
        converted = BufferFactory.convert(buffer, self.pixel_format)

        # Convert to a numpy array
        shape = converted.height, converted.width, int(converted.bits_per_pixel / 8)
        np_img = cv2.cvtColor(
            numpy.ctypeslib.as_array(converted.pdata, shape=shape).reshape(*shape),
            cv2.COLOR_BGR2RGB,
        )

        # Extract image metadata using chunks appended to buffer
        # NOTE: After requeue_buffer is called, accessing the values in
        # metadata_dict will cause a silent crash. Extract them here.
        try:
            metadata_dict = buffer.get_chunk(
                [f"Chunk{name}" for name in self.config["metadata"]]
            )
            metadata = {
                name.lower(): metadata_dict[f"Chunk{name}"].value
                for name in self.config["metadata"]
            }
        except ValueError:
            # Fail to get a chunk
            # TODO: Debug this! Contact support. I have a screenshot
            self.log.error(
                f"No Chunk data for an image! Camera ({self.ip}, {self.serial})"
            )
            metadata = {}

        # Reset the camera. It's important that we do cvtColor (or array.copy())
        # on the converted data before this line
        BufferFactory.destroy(converted)
        self.device.requeue_buffer(buffer)

        # Write the metadata
        metadata.update(
            {
                "completion-time": time.time(),
                "ip": self.ip,
                "camtype": self.config["camtype"],
                "serial": int(self.device_info["serial"]),
                "model": self.device_info["model"],
                "firmware-version": self.device_info["version"],
                "target-brightness": self.config["camera"].get(
                    "TargetBrightness", "unknown"
                ),
                "gamma": self.config["camera"].get("Gamma", "unknown"),
                "exposure-auto": self.config["camera"]["ExposureAuto"],
                "exposure-aoi-enable": self.config["camera"]["AutoExposureAOIEnable"],
                "gain-auto": self.config["camera"]["GainAuto"],
                "awb-auto": self.config["camera"].get("BalanceWhiteAuto", "unknown"),
                "awb-aoi-enable": self.config["camera"]["AwbAOIEnable"],
                "rgb-values": self.config.get("RGB", "unknown"),
                "aoi-values": self.config.get("AOI", "unknown"),
            }
        )

        # Finally, return
        return np_img, metadata

    def shutdown(self):
        self.device.stop_stream()
        system.destroy_device(self.device)


def video_thread(imdir, log, config, wipe=False, N=150):
    """Take N images and save them to the imdir as fast as possible."""

    assert N <= 600, "There's a RAM limit to the images we can store"

    try:

        log.info("Starting thread")

        # Set up the save directory
        if not imdir.is_dir():
            imdir.mkdir()

        di = system.device_infos
        assert len(di) == 1

        if wipe:
            camera = system.create_device(di[0])[0]
            camera.nodemap["DeviceFactoryReset"].execute()
            system.destroy_device()
            del camera
            di = system.device_infos
            while len(di) < 1:
                print("Waiting to reconnect after reset...")
                time.sleep(2)
                di = system.device_infos

        # Load camera config
        config["ip"] = di[0]["ip"]
        camera = ArenaCam(config, log)

        results = []
        input(f"Will capture {N} images when [Enter] is hit")
        print("Capturing -----")

        for _ in tqdm(range(N)):
            results.append(camera.grab_vidframe())

        print("Saving -----")
        for i, (image, metadata) in enumerate(tqdm(results)):
            path = imdir.joinpath(f"{i:04}.jpg")
            cv2.imwrite(str(path), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
            json.dump(
                metadata,
                path.with_suffix(".json").open("w"),
                indent=4,
                sort_keys=True,
            )

        camera.shutdown()
        log.info("Process done")

    except:
        log.exception("EXCEPTION OCCURRED")
        raise


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "-e",
        "--exposure",
        help="Exposure time in us (microseconds).",
        type=float,
        default=1000.0,
    )
    parser.add_argument(
        "-N",
        "--number-frames",
        help="Number of images to take.",
        type=int,
        default=250,
    )
    args = parser.parse_args()

    config = {
        "camtype": "color",
        "format": "BGR",
        "camera": {
            "DeviceStreamChannelPacketSize": 9000,  # Suggested value from Lucid support
            "AcquisitionFrameRateEnable": False,
            "ChunkModeActive": True,
            # "AcquisitionFrameRate": 0.8,
            "TriggerMode": "Off",
            # "TriggerSelector": "FrameStart",
            # "TriggerSource": "Software",
            "ExposureAuto": "Off",  # "Off" "Once" "Continuous"
            "ExposureAutoLimitAuto": "Off",  # "Off" "Once" "Continuous"
            "GainAuto": "Off",  # "Off" "Once" "Continuous"
            "BalanceWhiteAuto": "Off",  # "Off" "Once" "Continuous"
            "AwbAOIEnable": False,
            "AutoExposureAOIEnable": False,
            # NOTE: MUST be synced with the F-stop
            # "ExposureAutoLowerLimit": 300.0,
            "TargetBrightness": 20,
            "ExposureTime": args.exposure,
            "Gain": 0.0,
            "Gamma": 1.0,
        },
        "RGB": [1.45, 1.0, 2.4],
        "stream": {
            "StreamAutoNegotiatePacketSize": True,
            "StreamPacketResendEnable": True,
            "StreamBufferHandlingMode": "NewestOnly",
        },
        "metadata": ["ExposureTime", "Gain"],
    }

    logging.basicConfig(format="%(asctime)-16s %(message)s")
    log = logging.getLogger()
    log.setLevel(logging.INFO)
    video_thread(
        imdir=Path("/tmp/images/"), config=config, log=log, N=args.number_frames
    )
