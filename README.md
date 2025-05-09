# flow_videos
Analysis of videos of flow


## Data capture

```
cd capture/
```

### How to build Docker

```
docker build --build-arg UID=$(id -u) --build-arg GID=$(id -g) -t flow .
```

### How to run Docker

```
mkdir /tmp/images/
docker run -it --rm --net host -u $(id -u):$(id -g) --volume /tmp/images/:/tmp/images/ flow -e <exposure>
```

Here is an example usecase that can be used for repeated data gathering

```
export EXP=6000
export DIR=e_"$EXP"/l_95_3/
mkdir -p ~/Pictures/flowdata/$DIR; docker run -it --rm --net host -u $(id -u):$(id -g) --volume ~/Pictures/flowdata/$DIR:/tmp/images/ flow --exposure $EXP --number-frames 100
```

### Installing Arena

On your computer, follow these steps
* Follow the Arena SDK for Linux steps here: https://support.thinklucid.com/arena-sdk-documentation/
    * In order to “update ethernet drivers” I just did `sudo apt update; sudo apt full-upgrade`
    * Ignoring the Jumbo Frames steps, MTU of 9000 is over my hardware limit
    * Couldn’t run ethtool step “netlink error”
    * Had to modify the rmem steps
        * `sudo sh -c "echo 'net.core.rmem_default=134217728' >> /etc/sysctl.conf"`
        * `sudo sh -c "echo 'net.core.rmem_max=134217728' >> /etc/sysctl.conf"`
        * `sudo sysctl -p`
* Download and unpack the SDK: https://thinklucid.com/downloads-hub/
    * Arena SDK – x64 Ubuntu 22.04/24.04
    * `cp ~/Downloads/ArenaSDK_v0.1.95_Linux_x64.tar.gz /path/to/flow_videos/Arena/`
    * `tar -xzvf ArenaSDK...tar.gz; rm ArenaSDK...tar.gz`
        * Instructions are found in `Arena/README`
        * You can confirm with `g++ -v` that the version >5, and `make -v` should be installed
* Download and unpack the Area Python Package: https://thinklucid.com/downloads-hub/
    * `cp ~/Downloads/arena_api-2.7.1-py3-none-any.zip /path/to/flow_videos/Arena/python/`
    * `unzip arena_api...zip; rm arena_api...zip`


## Data analysis

### Data preparation

The first step is to take the video data (series of images) and turn them into clipped and processed windows, with stats calculated for each window. We want to calculate the stats once, then we can try to fit various stats against our state of interest in the training step. Example run of the preparation tool:

```
~/flow_videos$ python3 -m  assess.prepare.snip --data-dir /path/to/capture/ --start-stop range.json --lookup-dict state.json --windows windows.json --save-dir /tmp/ --downsample 20
```

The output will take the form
```
/tmp/
    data_<timestamp>/
        train/
            clipped image windows of the desired resolution
        test/
            clipped image windows of the desired resolution
        metadata.json
            {
                "kwargs": kwargs,
                "name": amalgamate trial name,
                {
                    path/to/image: {stats, origin image, origin window}
                    ...
                }
            }
```

### Training

TODO: We should be able to pass hyperparameters into the regressors

The next step is to train various basic regressors against the image stats calculated in data preparation. The models will be saved and can be assessed next. Example run of the training tool:

```
~/flow_videos$ python3 -m assess.train.train --data-dir path/to/data_<timestamp>/ --save-dir /tmp/ --stat <chosen stat> --state-key <state key> --model <model>
```

An example of a loop over various settings:

```
~/flow_videos$ for directory in path/to/data_*/; do
    for stat in gray-average hsv-average rgb-average; do
        for model in SVR ridge forest; do
            for flag in "--scale" ""; do
                python3 -m assess.train.train --save-dir /tmp/ --state-key <state key> --data-dir "$directory" --stat "$stat" --model "$model" $flag
            done
        done
    done
done
```

The output will take the form
```
/tmp/
    data_<input timestamp>_<created timestamp>/
        model.joblib
        metadata.json
            {
                "kwargs": kwargs,
                "name": amalgamate trial name,
                "R2_*", "RMSE_*", "MAE_*": training stats for model
                "X_*", "y_*", "y_pred_*": raw data and labels for model
            }
```

### Assessment

```
~/flow_videos$ python3 -m assess.assess.compare --data-dir /path/to/train/data/ --stats <stat> --plot-ranking --plot-vs-variable meta.kwargs.stat
```