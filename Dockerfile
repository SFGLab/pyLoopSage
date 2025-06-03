FROM nvidia/cuda:12.3.2-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

# Install Python 3.10, dev tools, and extras
RUN apt-get update && apt-get install -y \
    software-properties-common \
    wget curl git build-essential \
    libssl-dev zlib1g-dev libbz2-dev libreadline-dev \
    libsqlite3-dev libncursesw5-dev libffi-dev \
    liblzma-dev python3-pip python3-venv python3-dev \
    vim tmux tree htop less unzip \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Set Python 3.10 as default
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1 && \
    update-alternatives --install /usr/bin/pip pip /usr/bin/pip3 1

# Set workdir
WORKDIR /app

# Create virtual environment and install pyloopsage
RUN python -m venv /opt/pyloopsage-env && \
    /opt/pyloopsage-env/bin/pip install --upgrade pip setuptools && \
    /opt/pyloopsage-env/bin/pip install pyLoopSage

# Activate environment by default
ENV PATH="/opt/pyloopsage-env/bin:$PATH"

# Copy application files (if any)
COPY . /app

# Default shell
CMD ["/bin/bash"]