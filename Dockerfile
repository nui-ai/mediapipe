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
        software-properties-common && \
    apt-get update && apt-get install -y openjdk-21-jdk && \
    apt-get install -y mesa-common-dev libegl1-mesa-dev libgles2-mesa-dev && \
    apt-get install -y mesa-utils && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Remove broken Clang 16 install, use Ubuntu's clang instead
RUN apt-get update && apt-get install -y clang clang-format

RUN apt-get update && apt-get install -y python3-venv

# Create a virtual environment
RUN python3 -m venv /opt/venv

# Use the virtual environment for all subsequent commands
ENV PATH="/opt/venv/bin:$PATH"

# Copy Bazel configuration for C++17
COPY .bazelrc /mediapipe/.bazelrc

RUN pip install --upgrade pip setuptools wheel
RUN pip install future
RUN pip install absl-py "numpy<2" jax[cpu] opencv-contrib-python protobuf==3.20.1
RUN pip install six==1.14.0
RUN pip install tensorflow
RUN pip install tf_slim

RUN ln -s /usr/bin/python3 /usr/bin/python

# Install bazel
ARG BAZEL_VERSION=6.5.0
RUN mkdir /bazel && \
    wget --no-check-certificate -O /bazel/installer.sh "https://github.com/bazelbuild/bazel/releases/download/${BAZEL_VERSION}/b\
azel-${BAZEL_VERSION}-installer-linux-x86_64.sh" && \
    wget --no-check-certificate -O  /bazel/LICENSE.txt "https://raw.githubusercontent.com/bazelbuild/bazel/master/LICENSE" && \
    chmod +x /bazel/installer.sh && \
    /bazel/installer.sh  && \
    rm -f /bazel/installer.sh

COPY . /mediapipe/

# Fetch dependencies so XNNPACK is available for patching
#
# XNNPACK is a TensorFlow Lite dependency that sometimes tries to build microkernels
# for very new instruction sets (e.g., AVX-VNNI-INT8) that are not supported by all CPUs or GCC versions.
# This can cause build failures on CPUs (like Intel i7-14700) that do not support these instructions,
# or with compilers that do not recognize the relevant flags (e.g., -mavxvnniint8).
#
# We fetch dependencies here so we can patch the problematic build rules before the actual build.
RUN bazel fetch --experimental_repo_remote_exec //mediapipe/python/solutions:hands

# Patch XNNPACK BUILD.bazel files to remove avxvnniint8 kernels and flags
#
# The following commands search all BUILD.bazel files in the Bazel cache and remove any lines
# containing 'avxvnniint8' or the problematic '-mavxvnniint8' compiler flag. This prevents Bazel
# from trying to build those microkernels, which would otherwise cause the build to fail.
#
# This patch is necessary because XNNPACK is not vendored in our repository and is pulled in as
# an external dependency, so we cannot patch it directly in our codebase. Automating this step
# here ensures reproducible, hands-off builds.
RUN find /root/.cache/bazel -type f -name BUILD.bazel -exec sed -i '/avxvnniint8/d' {} +
RUN find /root/.cache/bazel -type f -name BUILD.bazel -exec sed -i '/-mavxvnniint8/d' {} +

# If we want the docker image to contain the pre-built object_detection_offline_demo binary, do the following
# RUN bazel build -c opt --define MEDIAPIPE_DISABLE_GPU=1 mediapipe/examples/desktop/demo:object_detection_tensorflow_demo
