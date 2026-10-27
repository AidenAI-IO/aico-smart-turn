# Conversion environment for exporting Smart Turn v3 to RKNN (Rockchip NPU).
#
# rknn-toolkit2 only ships Linux wheels (x86_64 and aarch64), so on macOS the
# conversion runs inside this container. Build it for linux/amd64: the aarch64
# wheel needs a real NPU to compile. See rknn/README.md for usage.
#
# Base image pinned by digest (multi-arch index of python:3.10-slim).
FROM python:3.10-slim@sha256:6ff506466f8b1e981719b468cc0023b8ef000b983cdec0eafe119d01f379e31d

RUN apt-get update \
    && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

WORKDIR /work
