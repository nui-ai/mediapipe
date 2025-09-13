#!/bin/bash
# Build script for CPU-only MediaPipe hand inference
# Usage: ./build_cpu_minimal.sh [clean|test|run]

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

TARGET="//mediapipe/examples/desktop/hand_tracking:hand_tracking_cpu_minimal"
CONFIG="--config=cpu_minimal"

case "${1:-build}" in
    clean)
        echo "Cleaning build cache..."
        bazel clean --expunge
        ;;
    test)
        echo "Building and testing CPU-only hand tracking..."
        bazel build ${CONFIG} ${TARGET}
        echo "Build completed successfully!"
        echo "Binary location: bazel-bin/mediapipe/examples/desktop/hand_tracking/hand_tracking_cpu_minimal"
        ;;
    run)
        echo "Building and running with webcam..."
        bazel build ${CONFIG} ${TARGET}
        echo "Starting hand tracking (press any key in window to exit)..."
        bazel-bin/mediapipe/examples/desktop/hand_tracking/hand_tracking_cpu_minimal \
            --calculator_graph_config_file=mediapipe/graphs/hand_tracking/hand_tracking_desktop_live.pbtxt
        ;;
    build|*)
        echo "Building CPU-only hand tracking..."
        bazel build ${CONFIG} \
            --copt=-march=native \
            --copt=-mtune=native \
            --copt=-O3 \
            ${TARGET}
        echo "Build completed!"
        echo "Binary: bazel-bin/mediapipe/examples/desktop/hand_tracking/hand_tracking_cpu_minimal"
        ;;
esac