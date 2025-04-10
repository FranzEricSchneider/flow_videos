# flow_videos
Analysis of videos of flow


## Data capture

```
cd capture/
```

### How to build Docker

```
docker build -t flow .
```

### How to run Docker

```
docker run -it --rm --net host --env="DISPLAY" --env="QT_X11_NO_MITSHM=1" --volume="/tmp/.X11-unix:/tmp/.X11-unix:rw" --volume="${XAUTHORITY}:/root/.Xauthority" --entrypoint /bin/bash flow
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

TODO: Create a data collection directory and Dockerfile
