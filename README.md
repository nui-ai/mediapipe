# MediaPipe v0.10.13 Build Guide 

This guide explains how to reproducibly build MediaPipe v0.10.13 for just the hand tracking target, at revision tag v0.10.13 of mediapipe which this forked repository was reverted to.
Judging from experience you need to work a few days to make it happen, as the build code will fail with modern Bazel, versions of dependencies it will fetch from the Internet which are not the same as when this build was originally working at the time of v0.10.13 release, and similar issues with its last-mile pip install for python proof of concept. Sometimes AI gets it right after just one day of careful iteration. 

- Success means:
  - A build command that's more specific than the original MediaPipe build instructions to avoid more errors from unnecessary build targets
  - Building a docker image to have the necessary OS dependencies for reproducibly running the build independently of the host system environment
  - Actually running the build inside a docker container using that docker image
  - Running the resulting Python verification script inside the container as well (if desired)

# Why this is never 100% future proof
The build process relies recursively on dependencies being fetched from over the internet. These dependencies may change, be removed, or otherwise become incompatible with the build process. Which is why we needed the changes comited on this forked repository, and why other changes may arise as necessary in the future.

# Guidelines for Stabilizing the Build Process

## Understanding the role of `bazel clean --expunge`

Use `bazel clean --expunge` before every build to ensure:

1. **Full reproducibility**: Each build starts from a pristine state
2. **Elimination of cached problems**: No lingering issues from previous builds, no false success which only works due to cached artifacts
3. **Validation of the entire dependency chain**: Confirms that all dependencies can be properly resolved

While this approach increases build time (as dependencies must be re-downloaded and rebuilt), it provides the highest level of confidence that your build process is reliable and reproducible. For development environments where quick iteration is needed, you can skip this step, but always return to a clean build before finalizing changes.

## Build Stabilization Recipe

To stabilize the MediaPipe build process, follow this step-by-step approach:

### 1. Stabilize the build on your Local Ubuntu Machine

```bash
# Build the specific target
bazel build -c opt --define MEDIAPIPE_DISABLE_GPU=1 --copt=-I/usr/include/opencv4 mediapipe/python/solutions:hands
```

If build errors occur:
1. Identify dependency issues in the WORKSPACE file
2. Fix one issue at a time
3. Test the build after each fix
4. Run `bazel clean --expunge` between major changes

### 2. Stabilize Docker Build which also uses Ubuntu as its base image

Once the direct build works:

1. Update the Dockerfile if necessary to match the environment where the direct build succeeded
2. Build the Docker image:
   ```bash
   docker build --no-cache -t mediapipe-build .
   ```
3. Test the build inside the container:
   ```bash
   docker run --rm -it -v "$PWD":/mediapipe mediapipe-build /bin/bash -c "cd /mediapipe && bazel build -c opt --define MEDIAPIPE_DISABLE_GPU=1 --copt=-I/usr/include/opencv4 mediapipe/python/solutions:hands"
   ```

### 4. Stabilize Python Package Installation

After the build succeeds:

1. Test the Python package build:
   ```bash
   python setup.py bdist_wheel
   ```
2. Address any pip/setuptools compatibility issues
3. Test installation in a virtual environment:
   ```bash
   python -m venv test_env
   source test_env/bin/activate
   pip install dist/mediapipe-*.whl
   ```
   - **Modern pip restrictions**: You will have to address the consequence of stricter dependency handling in newer pip versions, which were not an issue at the time of v0.10.13's release.

## Common Issues

The build issues arise from both modern versions of Bazel behaving a little differently and from more recent versions of dependencies pulled by Bazel, for dependencies which were previously not hard-pinned to specific versions. Common issues include:

- **Dependency cycles**: Often fixed by rearranging dependency declarations in WORKSPACE
- **Missing symbols**: May require specific versions of dependencies
- **Python compatibility**: Ensure compatibility with target Python versions

## Why this Recipe Matters

By following this structured approach:
1. We isolate build issues from dependency problems
2. We create a reproducible build process
3. We ensure both direct builds and containerized builds work consistently
4. We maintain compatibility with Python packaging systems

**Important**: Always start from the the v0.10.13 tag commit level of the originlal MediaPipe repository, which is our target version.

## Prerequisites

- **Docker** must be installed and ready for use on the host system.

## Build the Docker Image

From the project root (where the `Dockerfile` is located):

```bash
docker build --no-cache -t mediapipe-build .
```

- The `--no-cache` flag ensures all patches and updates are applied.

## Start an Interactive Container

Mount your project directory and start a shell:

```bash
docker run --rm -it -v "$PWD":/mediapipe mediapipe-build /bin/bash
```

- Your files are available at `/mediapipe` inside the container.

## Build MediaPipe Targets (Inside Container)

To build the Python hand landmarks solution (the only target built in the Docker image):

```bash
bazel build -c opt --define MEDIAPIPE_DISABLE_GPU=1 --copt=-I/usr/include/opencv4 mediapipe/python/solutions:hands
```

- This command builds only the hand landmarks solution for Python, matching the Docker image build step.
- The `--copt=-I/usr/include/opencv4` flag is needed for OpenCV 4.x on Ubuntu 24.04+.

> **Note:** Sandboxed Bazel builds (e.g., with `--sandbox_debug`) may fail due to upstream or environment issues. Use the regular build command above for reliable results.

## Build and Use the MediaPipe Python Solution (Inside Container)

To build and install the MediaPipe Python package using the already bazel built OpenCV (recommended for faster builds):

```bash
MEDIAPIPE_LINK_OPENCV=1 pip install . --use-pep517
```

- This will instruct the build to use your system OpenCV and skip building OpenCV from source.
- Make sure OpenCV and its development headers are installed in your environment.
- Do **not** use `--install-option` with pip, as it is not supported with modern builds.

## Run the Python Example (Inside Container)

To verify the built mediapipe with a video file (place `input.avi` in `mediapipe/python/`):

```bash
python3 mediapipe/python/verify.py
```

- Check `mediapipe/python/verified-detections.json` for results after running `verify.py`.

# Versioning

The Python package version is now automatically set at build time to `0.10.13+git.{commit_hash}` (where `{commit_hash}` is the current short git commit hash). This ensures every build is uniquely versioned and PEP 440 compliant. No manual version fix is needed.

# Pip Build Last Issue

**Deprecation Warning:**

Building MediaPipe using the legacy `setup.py bdist_wheel` mechanism is deprecated and will be removed in a future version of pip. To future-proof your build process, use the standardized build interface by running:

    pip install . --use-pep517

see [pip issue #6334](https://github.com/pypa/pip/issues/6334).

## Additional Notes

- The Docker image can build all MediaPipe targets; adjust Bazel build targets as needed.
- All patching and setup steps should be ultimately captured by a working Dockerfile for short-term reproducibility (up until the reasons for no 100% stability work for a long enough time again out there).
