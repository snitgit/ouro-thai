FROM nvidia/cuda:12.8.1-devel-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y \
    python3 python3-venv python3-pip git \
    && rm -rf /var/lib/apt/lists/*

RUN ln -sf /usr/bin/python3 /usr/bin/python

WORKDIR /workspace

COPY requirements.txt .
RUN pip install --no-cache-dir --break-system-packages -r requirements.txt

COPY . .

ENTRYPOINT ["python", "train_qlora.py"]
