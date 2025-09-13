# MediaPipe Hand Inference - Ubuntu 24.04 Setup Guide

This guide provides setup instructions for building and running the minimal CPU-only MediaPipe hand inference pipeline on Ubuntu 24.04.

## System Dependencies

### Install Bazel

```bash
# Install Bazel 6.x (required for MediaPipe)
sudo apt update
sudo apt install apt-transport-https curl gnupg
curl -fsSL https://bazel.build/bazel-release.pub.gpg | gpg --dearmor | sudo tee /etc/apt/trusted.gpg.d/bazel.gpg
echo "deb [arch=amd64] https://storage.googleapis.com/bazel-apt stable jdk1.8" | sudo tee /etc/apt/sources.list.d/bazel.list
sudo apt update
sudo apt install bazel-6.3.2
sudo ln -sf /usr/bin/bazel-6.3.2 /usr/bin/bazel
```

### Install System Libraries

```bash
# Essential development tools
sudo apt install build-essential

# OpenCV dependencies (CPU-only)
sudo apt install \
    libopencv-dev \
    libopencv-contrib-dev \
    pkg-config

# Additional libraries for MediaPipe
sudo apt install \
    libgflags-dev \
    libgoogle-glog-dev \
    libprotobuf-dev \
    protobuf-compiler \
    libeigen3-dev

# Optional: for webcam support
sudo apt install v4l-utils
```

### Python 3.12+ (if needed)

```bash
# Ubuntu 24.04 includes Python 3.12 by default
python3 --version  # Should be 3.12+

# Install pip if needed
sudo apt install python3-pip
```

## Building MediaPipe Hand Inference

### Clone Repository

```bash
git clone https://github.com/nui-ai/mediapipe.git
cd mediapipe
git checkout hand-cpu-only-build  # Switch to the CPU-only branch
```

### Build CPU-Only Hand Tracking

```bash
# Build the minimal CPU-only hand tracking binary
bazel build --config=cpu_minimal \
    //mediapipe/examples/desktop/hand_tracking:hand_tracking_cpu_minimal

# Alternative: Use the minimal workspace directly
bazel --bazelrc=.bazelrc.cpu_minimal build \
    //mediapipe/examples/desktop/hand_tracking:hand_tracking_cpu_minimal

# For production builds with maximum optimization
bazel build --config=cpu_minimal \
    --copt=-march=native \
    --copt=-mtune=native \
    --copt=-O3 \
    //mediapipe/examples/desktop/hand_tracking:hand_tracking_cpu_minimal
```

## Running Hand Inference

### With Video File

```bash
# Run hand tracking on a video file
bazel-bin/mediapipe/examples/desktop/hand_tracking/hand_tracking_cpu_minimal \
    --calculator_graph_config_file=mediapipe/graphs/hand_tracking/hand_tracking_desktop_live.pbtxt \
    --input_video_path=/path/to/your/video.mp4 \
    --output_video_path=/path/to/output/video.mp4
```

### With Webcam

```bash
# Run hand tracking with webcam (real-time)
bazel-bin/mediapipe/examples/desktop/hand_tracking/hand_tracking_cpu_minimal \
    --calculator_graph_config_file=mediapipe/graphs/hand_tracking/hand_tracking_desktop_live.pbtxt
```

## Performance Optimization

### CPU-Specific Optimizations

The build is optimized for your specific CPU architecture using `-march=native`. For maximum performance:

```bash
# Check your CPU features
cat /proc/cpuinfo | grep flags

# For Intel CPUs with AVX2/AVX512
export CC=gcc-11
export CXX=g++-11

# Rebuild with architecture-specific optimizations
bazel clean
bazel build --config=cpu_minimal \
    --copt=-mavx2 \
    --copt=-mfma \
    //mediapipe/examples/desktop/hand_tracking:hand_tracking_cpu_minimal
```

### Memory Usage

The minimal build significantly reduces memory usage by:
- Excluding GPU/CUDA libraries (~500MB saved)
- Using CPU-only TensorFlow Lite (~200MB saved)
- Removing unnecessary delegates and hardware accelerators

Typical memory usage: ~100-200MB (vs 1GB+ for full MediaPipe)

## Troubleshooting

### Build Errors

```bash
# Clean build cache
bazel clean --expunge

# Verify Bazel version
bazel version  # Should be 6.x

# Check for missing dependencies
sudo apt install --fix-missing
```

### Runtime Issues

```bash
# Webcam permission issues
sudo usermod -a -G video $USER
# Log out and log back in

# Missing model files
ls -la bazel-bin/mediapipe/examples/desktop/hand_tracking/
# Should contain .tflite model files
```

### Performance Issues

- Ensure you're using the optimized build: `--config=cpu_minimal`
- Check CPU usage: `htop` - should use multiple cores efficiently
- For better performance, use smaller input resolution or reduce frame rate

## Model Information

The minimal build uses these pre-trained models:
- **Palm Detection**: `palm_detection_full.tflite` (1.3MB)
- **Hand Landmarks**: `hand_landmark_full.tflite` (6.8MB)

Both models are optimized for CPU inference using XNNPACK acceleration.

## Next Steps

- [MediaPipe Hand Tracking Guide](https://developers.google.com/mediapipe/solutions/vision/hand_landmarker)
- [Custom Model Training](https://developers.google.com/mediapipe/solutions/customization)
- [Performance Benchmarking](https://developers.google.com/mediapipe/framework/framework_concepts/benchmarking)