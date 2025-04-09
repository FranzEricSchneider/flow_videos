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

# Create user and grant sudo access
RUN useradd -ms /bin/bash user && \
    apt-get update && \
    apt-get install -y sudo --option=Dpkg::Options::="--force-confdef" --option=Dpkg::Options::="--force-confold" && \
    usermod -aG sudo user && \
    echo "user ALL=(ALL) NOPASSWD:ALL" > /etc/sudoers.d/user && \
    chmod 0440 /etc/sudoers.d/user
USER user
RUN mkdir /home/user/app
WORKDIR /home/user/app

# Create and activate virtual environment
RUN python3 -m venv /home/user/venv
ENV PATH="/home/user/venv/bin:${PATH}"
ENV VIRTUAL_ENV="/home/user/venv"

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

# Copy src directory into the container
COPY --chown=user:user src/ /home/user/app/src/
COPY --chown=user:user Arena/ /home/user/app/Arena/

# Switch back to root for a sudo command
USER root
RUN cd /home/user/app/Arena/ArenaSDK_Linux_x64/ && \
    sh Arena_SDK_Linux_x64.conf
USER user
RUN cd /home/user/app/Arena/python/ && \
    pip install arena_api-2.7.1-py3-none-any.whl

# # Install the source code as a package
# RUN pip3 install --user -e .
