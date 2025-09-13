#!/bin/bash
# Validation script for MediaPipe CPU-only setup on Ubuntu 24.04

set -e

echo "=== MediaPipe CPU-Only Setup Validation ==="
echo

# Check OS version
if [ -f /etc/os-release ]; then
    . /etc/os-release
    echo "✓ OS: $PRETTY_NAME"
    if [[ "$VERSION_ID" != "24.04" ]]; then
        echo "⚠ Warning: This setup is optimized for Ubuntu 24.04"
    fi
else
    echo "⚠ Cannot determine OS version"
fi

# Check Bazel
if command -v bazel &> /dev/null; then
    BAZEL_VERSION=$(bazel --version 2>/dev/null | head -n1 || echo "unknown")
    echo "✓ Bazel: $BAZEL_VERSION"
    if ! bazel --version 2>/dev/null | grep -E "bazel [3-6]\." > /dev/null; then
        echo "⚠ Warning: Bazel 3.x-6.x recommended for MediaPipe"
    fi
else
    echo "✗ Bazel not found - install with:"
    echo "   sudo apt install bazel-6.3.2"
fi

# Check essential development tools
echo
echo "=== Development Tools ==="
for tool in gcc g++ make pkg-config; do
    if command -v $tool &> /dev/null; then
        echo "✓ $tool: $(which $tool)"
    else
        echo "✗ $tool not found"
    fi
done

# Check OpenCV
echo
echo "=== OpenCV Libraries ==="
if pkg-config --exists opencv4; then
    echo "✓ OpenCV4: $(pkg-config --modversion opencv4)"
elif pkg-config --exists opencv; then
    echo "✓ OpenCV: $(pkg-config --modversion opencv)"
else
    echo "✗ OpenCV not found - install with:"
    echo "   sudo apt install libopencv-dev"
fi

# Check other required libraries
echo
echo "=== Required Libraries ==="
declare -A libs=(
    ["protobuf"]="libprotobuf-dev"
    ["eigen3"]="libeigen3-dev"  
    ["glog"]="libgoogle-glog-dev"
    ["gflags"]="libgflags-dev"
)

for lib in "${!libs[@]}"; do
    if pkg-config --exists $lib; then
        echo "✓ $lib: $(pkg-config --modversion $lib)"
    else
        echo "✗ $lib not found - install with:"
        echo "   sudo apt install ${libs[$lib]}"
    fi
done

# Check Python (optional)
echo
echo "=== Python Environment ==="
if command -v python3 &> /dev/null; then
    PYTHON_VERSION=$(python3 --version 2>&1 | cut -d' ' -f2)
    echo "✓ Python: $PYTHON_VERSION"
    if [[ "$PYTHON_VERSION" < "3.12" ]]; then
        echo "⚠ Warning: Python 3.12+ recommended"
    fi
else
    echo "⚠ Python3 not found (optional for this build)"
fi

# Check webcam (optional)
echo
echo "=== Webcam Support ==="
if ls /dev/video* &> /dev/null; then
    echo "✓ Video devices found: $(ls /dev/video* | tr '\n' ' ')"
else
    echo "⚠ No video devices found (webcam optional for testing)"
fi

# Check MediaPipe files
echo
echo "=== MediaPipe Files ==="
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

files=(
    "WORKSPACE.cpu_minimal"
    ".bazelrc.cpu_minimal"
    "mediapipe/examples/desktop/hand_tracking/BUILD.cpu_minimal"
    "mediapipe/examples/desktop/hand_tracking/hand_tracking_minimal_main.cc"
    "third_party/tensorflow_lite_cpu.BUILD"
)

for file in "${files[@]}"; do
    if [ -f "$SCRIPT_DIR/$file" ]; then
        echo "✓ $file"
    else
        echo "✗ Missing: $file"
    fi
done

echo
echo "=== Summary ==="
echo "Run './build_cpu_minimal.sh test' to test the build"
echo "Run './build_cpu_minimal.sh run' to test with webcam"
echo "See docs/ubuntu_24_04_setup.md for detailed instructions"