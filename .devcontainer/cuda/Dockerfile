FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

# NVIDIA base image is minimal; install the tools required by common-utils and
# subsequent devcontainer features. Everything else (uv, zsh, gh, claude-code)
# is layered on via features in devcontainer.json.
RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates curl git sudo openssh-client tzdata locales \
    && locale-gen en_US.UTF-8 \
    && rm -rf /var/lib/apt/lists/*

ENV LANG=en_US.UTF-8 LC_ALL=en_US.UTF-8
