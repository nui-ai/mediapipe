# MediaPipe Hand Tracking Rust Runner

The Rust project directory in this repo provides a runner for the MediaPipe hand tracking pipeline using the project's C API via FFI.

## Prerequisites
- Rust and Cargo (already installed)
- OpenCV (system libraries)
- Bazel (to build the C API shared library)

## C API Shared Library

You must build the C API shared library using Bazel. For example, in your Bazel BUILD file (`mediapipe/examples/desktop/BUILD`):

```bazel
cc_binary(
    name = "libhands_pipeline_operator_c_api.so",
    ...,
    linkshared = 1,
    visibility = ["//visibility:public"],
)
```

This will produce `libhands_pipeline_operator_c_api.so` in:
`bazel-bin/mediapipe/examples/desktop/hand_tracking/`

## How Rust Uses the Shared Library

- **Build time:**
  - The Rust build script (`build.rs`) tells Cargo/rustc to add the Bazel output directory to the linker search path, so it can find `libhands_pipeline_operator_c_api.so` when building your Rust binary.
- **Run time:**
  - You must set the `LD_LIBRARY_PATH` environment variable to include the Bazel output directory so the dynamic loader can find the `.so` file when running your Rust binary.
  - Example:
    ```bash
    export LD_LIBRARY_PATH=$PWD/../bazel-bin/mediapipe/examples/desktop/hand_tracking:$LD_LIBRARY_PATH
    ```
- **No additional Rust code is needed to load the library;** the `extern "C"` declarations in `ffi.rs` are sufficient for FFI calls.

## Building the Rust Project

From the `rust/` directory:

```bash
cargo build --release
```

## Running the Rust Pipeline Runner

Set the `LD_LIBRARY_PATH` as described above, then run the Rust binary:

```bash
cargo run --release -- \
  --graph-file ../mediapipe/modules/hand_landmark/hand_landmark_tracking_cpu.pbtxt \
  --input-video-path ../sample-video.avi
```

- `--graph_file` is the path to your MediaPipe graph config.
- `--input_video_path` is optional (if omitted, uses camera).
- Output protobuf will be written to `output_data_cpp.pb` in the current directory.

## Notes
- Make sure OpenCV is installed and available to both Bazel and Rust.
- The Rust binary uses FFI to call the C API in the shared library.
- If you change the C API, re-build the `.so` file before running Rust.

## Troubleshooting
- If you get errors about missing symbols or libraries, check both the linker search path in `build.rs` and your `LD_LIBRARY_PATH`.
- If you change the protobuf definition, re-run `cargo clean && cargo build` to regenerate Rust code.

## System OpenCV and Compiler Prerequisites

This project links to your system OpenCV installation, just like your C++ Bazel builds. You do NOT need to build OpenCV from source for Rust.

- **Make sure you already have these installed:**
  ```bash
  sudo apt-get update
  sudo apt-get install libopencv-dev pkg-config
  ```
  This provides headers in `/usr/include/opencv4` and shared libraries in `/usr/lib`.

- **Install LLVM and Clang (required for Rust OpenCV bindings):**
  ```bash
  sudo apt-get install llvm clang libclang-dev
  ```
  This does not change your system's default C/C++ compiler or affect Bazel builds. It only enables Rust's `opencv` crate to generate bindings at build time.

- **Why are LLVM and Clang required for Rust's OpenCV crate?**
  The Rust `opencv` crate does not ship with pre-generated bindings. Instead, it generates FFI bindings to your system's OpenCV installation at build time using `libclang` and LLVM. This ensures the Rust bindings match the exact OpenCV version and configuration on your machine, avoiding ABI mismatches and maximizing compatibility. Pre-generated bindings would not be portable or reliable across different systems and OpenCV versions.

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

- **Summary:**
  - Rust and Bazel will both link to the system OpenCV installation.
  - Installing `llvm`, `clang`, and `libclang-dev` is required for Rust's OpenCV crate, but should not affect your any other C++ build setup.
  - No changes to your Bazel or C++ compiler configuration are needed for the Rust build.
