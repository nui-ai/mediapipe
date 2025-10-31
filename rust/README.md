# MediaPipe Hand Tracking Rust Runner

The Rust project directory in this repo provides a runner for the MediaPipe hand tracking pipeline using the project's C API via FFI.

## Prerequisites
1. Rust and Cargo (already installed)
2. `llvm`, `clang`, and `libclang-dev` is required for Rust's OpenCV crate, but should not affect your any other C++ build setup:
   - **Why are LLVM and Clang required for Rust's OpenCV crate?**
     The Rust `opencv` crate does not ship with pre-generated bindings. Instead, it generates FFI bindings to your system's OpenCV installation at build time using `libclang` and LLVM. This ensures the Rust bindings match the exact OpenCV version and configuration on your machine, avoiding ABI mismatches and maximizing compatibility. Pre-generated bindings would not be portable or reliable across different systems and OpenCV versions.
   - **Install LLVM and Clang (required for Rust OpenCV bindings):**
     ```bash
     sudo apt-get install llvm clang libclang-dev
     ```
     This does not change your system's default C/C++ compiler or affect Bazel builds. It only enables Rust's `opencv` crate to generate bindings at build time.
3. Bazel must have run its build target producing a shared library of our mediapipe C API wrapper:  

## Mediapipe Wrapper C API Shared Library

a C API shared library using Bazel. For example, in your Bazel BUILD file (`mediapipe/examples/desktop/BUILD`):

+ Bazel builds `libhands_pipeline_operator_c_api.so` via a cc_binary build target names as such.
+ The rust build consumes it from bazel's output directory. 
+ Specifically, the Rust build script (`build.rs`) tells Cargo/rustc to add the Bazel output directory to the linker search path, so it can find `libhands_pipeline_operator_c_api.so` when building the Rust code.
+ The `extern "C"` declarations present in `ffi.rs` are then enough for enabling calling its functions from rust. 

## Building the Rust Project

From the `rust/` directory:

```bash
cargo build --release
```

## Running the Rust Pipeline Runner

```bash
cargo run --release -- \
  --graph-file mediapipe/modules/hand_landmark/hand_landmark_tracking_cpu.pbtxt \
  --input-video-path ../sample-video.avi
```

- `--graph_file` is the path to your MediaPipe graph config.
- `--input_video_path` is optional (if omitted, uses camera).
- Output protobuf will be written to `output_data_cpp.pb` in the current directory.

---

### System OpenCV and Compiler Prerequisites ― obsolete as we link to our bazel built version of OpenCV now which solves https://github.com/nui-ai/mediapipe/issues/22

This project links to your system OpenCV installation, just like your C++ Bazel builds. You do NOT need to build OpenCV from source for Rust.

- **Make sure you already have these installed:**
  ```bash
  sudo apt-get update
  sudo apt-get install libopencv-dev pkg-config
  ```
  This provides headers in `/usr/include/opencv4` and shared libraries in `/usr/lib`.

- **Verify OpenCV is discoverable:**
  ```bash
  pkg-config --modversion opencv4
  ```
  This should print your OpenCV version if everything is set up correctly. It should be available already if you already set-up the mediapipe C++ project to successfully build, as it too links to your system OpenCV.

- **Environment variables (only if needed):**
  If your OpenCV or pkgconfig are installed in a non-standard location, set these before building, but it shouldn't typically be in a non-standard location:
  ```bash
  export PKG_CONFIG_PATH=/usr/lib/pkgconfig
  export OPENCV_INCLUDE_PATH=/usr/include/opencv4
  export OPENCV_LIB_PATH=/usr/lib
  ```

