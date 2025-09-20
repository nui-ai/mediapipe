#
# docker image for this repository's objective of building the mediapipe hands solution (build target) from scratch, specifically in a Ubuntu 24.04 container.
# it includes a step readying it to serve as a container for running a github self-hosted runner that can execute workflows
# for github copilot, since the github cloud environment gives copilot an impossibly hard time in network restrictions
# and configuration which is prohibitive to working with a complex bazel build that needs to pull a lot of dependencies
# from the internet.
#
# so you can use this image as such a container for a self-hosted github workflows runner for developing that building to work,
# or just as one that can build mediapipe from source (when that goal will have been accoplished).
#

# Copyright 2019 The MediaPipe Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

FROM ubuntu:24.04

LABEL maintainer="matan@nui.ai"

WORKDIR /io
WORKDIR /mediapipe

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc g++ \
    ca-certificates \
    curl \
    ffmpeg \
    git \
    wget \
    unzip \
    nodejs \
    npm \
    python3-dev \
    python3-opencv \
    python3-pip \
    libopencv-core-dev \
    libopencv-highgui-dev \
    libopencv-imgproc-dev \
    libopencv-video-dev \
    libopencv-calib3d-dev \
    libopencv-features2d-dev \
    software-properties-common \
    tar \
    openjdk-21-jdk \
    protobuf-compiler \
    libprotobuf-dev \
    mesa-common-dev \
    libegl1-mesa-dev \
    libgles2-mesa-dev \
    mesa-utils \
    libopenexr-dev \
    libimath-dev \
    libavcodec-dev \
    libavformat-dev \
    libavutil-dev \
    libswscale-dev \
    libswresample-dev \
    libdc1394-dev \
    libjpeg-dev \
    libpng-dev \
    libtiff-dev \
    pkg-config \
    cmake \
    clang \
    clang-format \
    python3-venv \
    && rm -rf /var/lib/apt/lists/*

# Copy Bazel configuration for C++17
COPY .bazelrc /mediapipe/.bazelrc

# Install bazel
ARG BAZEL_VERSION=6.5.0
RUN mkdir /bazel && \
    wget --no-check-certificate -O /bazel/installer.sh "https://github.com/bazelbuild/bazel/releases/download/${BAZEL_VERSION}/bazel-${BAZEL_VERSION}-installer-linux-x86_64.sh" && \
    wget --no-check-certificate -O  /bazel/LICENSE.txt "https://raw.githubusercontent.com/bazelbuild/bazel/master/LICENSE" && \
    chmod +x /bazel/installer.sh && \
    /bazel/installer.sh  && \
    rm -f /bazel/installer.sh

# Include the repository's code directory, this is actually too much if e.g. you have locally built mediapipe in the same directory, thus many build files will be included.
COPY . /mediapipe/

# Build mediapipe ― this makes sure that the image contains all the bazel and pip dependencies needed to build mediapipe, future proofing them inside the image.
# maybe upgrading pip, setuptools and wheel is overdoing it, as it worked without it, and upgrading to the latest Internet available versions may not help here.
RUN \
    cd /mediapipe && \
    git config --global --add safe.directory /mediapipe && \
    python3 -m venv .venv && \
    .venv/bin/python --version && \
    .venv/bin/pip --version && \
    .venv/bin/pip install --upgrade pip setuptools wheel && \
    .venv/bin/pip install --verbose .

# Don't try to install the Github actions runner, which is easy from the github page instructions, but just ready a user for it to use as it will refuse to start as root:
RUN RUN adduser --disabled-password --gecos "" github-runner

# Dedicated non-root user and directory for GitHub Actions runner; this failed entirely the other day so parking it.
# Also, running within a container is it necessary not to run as root within the container itself these days?
#RUN useradd -m runner && \
#    mkdir -p /github-runner && \
#    chown runner:runner /github-runner
#
#USER runner
#WORKDIR /github-runner

# Download and extract the github actions runner as non-root user in /github-runner, for the case that we want copilot agent mode (or other github workflows) to use a self-hosted local github runner within this container.
# This will only be useful as long as that runner version is supported by github, and otherwise just install the latest version of it as per https://github.com/nui-ai/mediapipe/settings/actions/runners/new,
# or as per the github instructions from its runners page's "New self-hosted runner" button, the page is currently at https://github.com/nui-ai/mediapipe/settings/actions/runners.
#ENV RUNNER_VERSION=2.328.0
#RUN cd /github-runner && \
#    curl -o actions-runner-linux-x64-${RUNNER_VERSION}.tar.gz -L https://github.com/actions/runner/releases/download/v${RUNNER_VERSION}/actions-runner-linux-x64-${RUNNER_VERSION}.tar.gz && \
#    echo "01066fad3a2893e63e6ca880ae3a1fad5bf9329d60e77ee15f2b97c148c3cd4e  actions-runner-linux-x64-${RUNNER_VERSION}.tar.gz" | shasum -a 256 -c && \
#    tar xzf ./actions-runner-linux-x64-${RUNNER_VERSION}.tar.gz && \
#    rm actions-runner-linux-x64-${RUNNER_VERSION}.tar.gz
