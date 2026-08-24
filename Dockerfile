# syntax=docker/dockerfile:1

ARG PYTHON_VERSION=3.12
FROM python:${PYTHON_VERSION}-slim-bookworm AS base

WORKDIR /app

COPY requirements.txt ./
RUN python -m pip install --upgrade pip \
    && python -m pip install -r requirements.txt

RUN addgroup --system benchmark \
    && adduser --system --ingroup benchmark --home /app benchmark \
    && chown benchmark:benchmark /app

COPY --chown=benchmark:benchmark src ./src

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    MPLCONFIGDIR=/tmp/matplotlib \
    NUMBA_CACHE_DIR=/tmp/numba-cache \
    PYTHONPATH=/app/src

USER benchmark
ENTRYPOINT ["python", "src/benchmark.py"]

FROM base AS cpu

ENV HOME=/tmp/benchmark \
    USER=benchmark

FROM base AS cuda

USER root
ARG PYTORCH_INDEX_URL=https://download.pytorch.org/whl/cu128
COPY requirements-cuda.txt ./
RUN python -m pip install \
        torch==2.11.0 \
        torchvision==0.26.0 \
        torchaudio==2.11.0 \
        --index-url "${PYTORCH_INDEX_URL}" \
    && python -m pip install -r requirements-cuda.txt
ENV HOME=/tmp/benchmark \
    USER=benchmark
USER benchmark
