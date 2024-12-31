# Stage 1: Build NGINX gateway
FROM nginx:latest AS gateway

# Copy the NGINX configuration file to the appropriate directory
COPY nginx.conf /etc/nginx/nginx.conf

# Stage 2: Build the application with NVIDIA CUDA base
FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04 AS backend

# Set environment variables to avoid interactive prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive

# Enable BuildKit-style caching for apt and install necessary packages
RUN --mount=type=cache,sharing=locked,target=/var/cache/apt \
    --mount=type=cache,sharing=locked,target=/var/lib/apt \
    set -eux; \
    apt-get update; \
    apt-get install -y --no-install-recommends \
        python3 \
        python3-pip \
        python3-venv \
        build-essential \
        libssl-dev \
        libffi-dev \
    ; \
    apt-get clean; \
    rm -rf /var/lib/apt/lists/*

# Set the working directory in the container
WORKDIR /agent

# Copy only requirements.txt first to leverage Docker layer caching
COPY requirements.txt /agent/

# Use BuildKit caching for pip to speed up builds during dependency installation
RUN --mount=type=cache,target=/root/.cache/pip \
    pip3 install --no-cache-dir -r requirements.txt

# Copy the application source code
COPY talent_faq_agent/ /python_code/talent_faq_agent/
