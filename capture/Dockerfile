# Start with the latest Ubuntu
FROM ubuntu:24.04

# Set environment variables to non-interactive (this prevents some prompts)
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y \
    eog \
    git \
    ipython3 \
    python3 \
    python3-pip \
    python3-venv \
    python3-dev \
    python3-full \
    build-essential \
    libgtk2.0-dev \
    pkg-config \
    libavcodec-dev \
    libavformat-dev \
    libswscale-dev \
    libtbb-dev \
    libjpeg-dev \
    libpng-dev \
    libtiff-dev \
    libdc1394-dev \
    libx264-dev \
    libx265-dev \
    screen \
    wget \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Accept UID and GID from build args
ARG UID=1000
ARG GID=1000

# Create user and grant sudo access
RUN apt-get update && \
    apt-get install -y sudo --option=Dpkg::Options::="--force-confdef" --option=Dpkg::Options::="--force-confold" && \
    usermod -aG sudo ubuntu && \
    echo "ubuntu ALL=(ALL) NOPASSWD:ALL" > /etc/sudoers.d/ubuntu && \
    chmod 0440 /etc/sudoers.d/ubuntu
USER ubuntu
RUN mkdir /home/ubuntu/app
WORKDIR /home/ubuntu/app

# Create and activate virtual environment
RUN python3 -m venv /home/ubuntu/venv
ENV PATH="/home/ubuntu/venv/bin:${PATH}"
ENV VIRTUAL_ENV="/home/ubuntu/venv"

# Install Python packages globally
RUN python3 -m pip install --upgrade pip

RUN pip3 install --no-cache-dir \
    imageio \
    numpy \
    matplotlib \
    opencv-python \
    pyyaml \
    scikit-learn \
    scipy \
    torch \
    torchsummaryX \
    torchvision \
    tqdm \
    wandb

# Install the Arena software for Lucid cameras
COPY --chown=ubuntu:ubuntu Arena/ /home/ubuntu/app/Arena/

# Switch back to root for a sudo command
USER root
RUN cd /home/ubuntu/app/Arena/ArenaSDK_Linux_x64/ && \
    sh Arena_SDK_Linux_x64.conf
USER ubuntu
RUN cd /home/ubuntu/app/Arena/python/ && \
    pip install arena_api-2.7.1-py3-none-any.whl

# Copy src directory into the container
COPY --chown=user:ubuntu src/ /home/ubuntu/app/src/
RUN mkdir -p /tmp/images

# Set up the entrypoint so image capture happens right away
ENTRYPOINT ["python3", "/home/ubuntu/app/src/cameras/arena.py"]
CMD []

# # Install the source code as a package
# RUN pip3 install --user -e .
