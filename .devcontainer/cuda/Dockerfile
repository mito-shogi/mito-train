FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

ARG USERNAME=vscode
ARG USER_UID=1000
ARG USER_GID=$USER_UID

# Keep BuildKit's apt cache mounts populated across builds: docker-clean wipes
# the archives after every install, so drop it and tell apt to keep them.
RUN rm -f /etc/apt/apt.conf.d/docker-clean \
    && echo 'Binary::apt::APT::Keep-Downloaded-Packages "true";' \
        > /etc/apt/apt.conf.d/keep-cache

# NVIDIA base image is minimal; install the tools required by common-utils and
# subsequent devcontainer features. Everything else (uv, zsh, gh, claude-code)
# is layered on via features in devcontainer.json.
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt/lists,sharing=locked \
    DEBIAN_FRONTEND=noninteractive apt-get update \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        ca-certificates curl git sudo openssh-client tzdata locales \
    && locale-gen en_US.UTF-8

ENV LANG=en_US.UTF-8 LC_ALL=en_US.UTF-8

# Create the non-root vscode user the devcontainer runs as (remoteUser). The
# NVIDIA base is root-only, so unlike the devcontainers base image we make it
# here with passwordless sudo.
RUN groupadd --gid $USER_GID $USERNAME \
    && useradd --uid $USER_UID --gid $USER_GID -m $USERNAME -s /bin/bash \
    && echo "$USERNAME ALL=(ALL) NOPASSWD:ALL" > /etc/sudoers.d/$USERNAME \
    && chmod 0440 /etc/sudoers.d/$USERNAME

USER $USERNAME
