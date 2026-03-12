FROM pytorch/pytorch:2.5.1-cuda12.1-cudnn9-devel

ENV PYTHONUNBUFFERED=1
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

RUN pip install --no-cache-dir --upgrade pip setuptools && \
    pip install --no-cache-dir -e sam2/ && \
    pip install --no-cache-dir -r api/requirements.txt

RUN cd sam2/checkpoints && chmod +x download_ckpts.sh && ./download_ckpts.sh

EXPOSE 8000

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
