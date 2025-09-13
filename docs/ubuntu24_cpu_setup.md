# minimal cpu-only build setup for ubuntu 24.04

this guide covers setting up mediapipe for cpu-only hand inference on ubuntu 24.04.

## dependency compatibility updates (2024-09-13)

**critical fixes applied:**
- ✅ abseil-cpp updated from future-dated `20250814.0` to stable `20240116.2`
- ✅ rules_cc updated from ancient `0.0.1` to current `0.0.9`
- ✅ bazel_skylib updated from `1.3.0` to current `1.5.0`  
- ✅ protobuf updated from `3.19.1` to more recent `23.4`
- ✅ zlib url changed from insecure http to https with github fallback
- ✅ removed blocked tensorflow mirror urls

## system dependencies

install required system packages:

```bash
sudo apt update
sudo apt install -y \
    build-essential \
    clang \
    cmake \
    git \
    python3 \
    python3-dev \
    python3-pip \
    pkg-config \
    zip \
    unzip \
    wget \
    curl \
    libopencv-dev \
    libopencv-contrib-dev
```

## bazel installation

**note:** `releases.bazel.build` is currently blocked. use github actions setup or alternative installation:

install bazel 6.5.0 via apt repository:

```bash
curl -fsSL https://bazel.build/bazel-release.pub.gpg | gpg --dearmor >bazel-archive-keyring.gpg
sudo mv bazel-archive-keyring.gpg /usr/share/keyrings
echo "deb [arch=amd64 signed-by=/usr/share/keyrings/bazel-archive-keyring.gpg] https://storage.googleapis.com/bazel-apt stable jdk1.8" | sudo tee /etc/apt/sources.list.d/bazel.list
sudo apt update && sudo apt install -y bazel-6.5.0
```

verify installation:

```bash
bazel version
```

## build configuration

the minimal cpu-only build excludes:
- gpu acceleration (opencl, metal, vulkan)
- cuda support
- edge tpu support  
- hardware delegates except xnnpack

only the following tensorflow lite components are included:
- cpu inference engine
- xnnpack delegate for optimized cpu performance
- builtin operations
- flatbuffers 2.0.0 for model serialization

this results in significantly smaller binary size and eliminates gpu driver dependencies.

## build instructions

clone the repository and switch to the cpu-only branch:

```bash
git clone https://github.com/nui-ai/mediapipe.git
cd mediapipe
git checkout hand-cpu-only-build
```

build the minimal hand tracking target:

```bash
bazel build --config=cpu-only -c opt \
    //mediapipe/examples/desktop/hand_tracking:hand_tracking_cpu_minimal
```

alternatively, build with explicit flags:

```bash
bazel build -c opt --define MEDIAPIPE_DISABLE_GPU=1 \
    --copt=-I/usr/include/opencv4 \
    //mediapipe/examples/desktop/hand_tracking:hand_tracking_cpu_minimal
```

## run the example

```bash
bazel-bin/mediapipe/examples/desktop/hand_tracking/hand_tracking_cpu_minimal \
    --calculator_graph_config_file=mediapipe/graphs/hand_tracking/hand_tracking_desktop_live.pbtxt \
    --input_video_path=<input_video.mp4> \
    --output_video_path=<output_video.mp4>
```

## tensorflow dependency

this build uses tensorflow 2.13.0 with:
- flatbuffers 2.0.0
- xnnpack delegate for cpu optimization
- cpu-only tflite inference

## troubleshooting

if you encounter build errors:
1. ensure opencv4 headers are in `/usr/include/opencv4`
2. verify bazel version matches `.bazelversion` file
3. check that `MEDIAPIPE_DISABLE_GPU=1` is set
4. verify internet connectivity to github.com for dependency downloads

**updated dependencies should resolve c++ header compatibility issues with modern compilers.**