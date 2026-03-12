FROM pytorch/pytorch:2.5.1-cuda12.1-cudnn9-devel

ENV PYTHONUNBUFFERED=2
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libavutil-dev \
    libavcodec-dev \
    libavformat-dev \
    libswscale-dev \
    pkg-config \
    build-essential \
    libffi-dev \
    wget \
    python3-opencv && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace
COPY . .

RUN pip install --upgrade pip setuptools && \
    pip install -r requirements.txt && \
    pip install hydra-core iopath decord

RUN cd sam2/checkpoints && chmod +x download_ckpts.sh && ./download_ckpts.sh

CMD ["/bin/bash"]
