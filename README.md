# MediaPipe v0.10.13 Build Guide 

This guide explains how to work towards reproducibly building MediaPipe v0.10.13 for just the hand tracking target, at revision tag v0.10.13 of mediapipe which this forked repository was reverted to.
Judging from experience you need to work a few days to make it happen, as the build code will fail with modern Bazel, versions of dependencies it will fetch from the Internet which are not the same as when this build was originally working at the time of v0.10.13 release, and similar issues with its last-mile pip install for python proof of concept. Sometimes AI gets it right after just one day of careful iteration.

This repository is also a codebase where a lot of voodoo took place on a first two-day run of this all ― but it contains the right build target commands named below, and a lean verification script `verify.py` that can be used to verify the build result when you get that far.

Techically it's a fork of the original MediaPipe repository, reverted to the v0.10.13 tag commit, with some patches applied to make it buildable again which didn't go all the way and should be restarted from v0.10.13 from scratch.

This repository would be between a prerequisite and a starting point for deriving a mediapipe-framework liberated C++ implementation of the hand tracking pipeline, which is the topic of the sibling repository [hand-tracking-cpp](https://github.com/nui-ai/mediapipe-liberation)

- Success means:
  - A build command that's more specific than the original MediaPipe build instructions to avoid more errors from unnecessary build targets
  - Building a docker image to have the necessary OS dependencies for reproducibly running the build independently of the host system environment
  - Actually running the build inside a docker container using that docker image
  - Running the resulting Python verification script inside the container as well (if desired)

# Why this is never 100% future proof
- The build process relies recursively on dependencies being fetched from over the internet. These dependencies may change, be removed, or otherwise become incompatible with the build process. Which is why we needed the changes comited on this forked repository, and why other changes may arise as necessary in the future. 
- We can pin down specific versions all across and hope they keep served on the Internet for long enough, to change the tradeoff a little.

# Why this is hard
Its a tug of war between old versions that can no longer install or work against the Internet repositories as they are today, and new versions that don't like elements of how the build (which was coded to old versions) is. this statement applies to both the Bazel and the pip parts in equal amounts!

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

Stabilizing it without docker means faster turn-over times.

```bash
# Build the specific target
bazel build -c opt --define MEDIAPIPE_DISABLE_GPU=1 --copt=-I/usr/include/opencv4 mediapipe/python/solutions:hands
```

You may choose to skip this step if you prefer to go directly to the Docker environment. Maybe that's a good idea. 

### 2. Stabilize a Docker image that succeeds in building the target

Once a local build works, or if you think your local machine is dirty or just prefer to skip it:

A Dockerfile already exists in this repository, which you can modify as needed or start from scratch.

1. Different versions of Ubuntu can play out differently in which versions of what they agree to install ― meaning different complexities, different bugs.
2. To build the Docker image:
   ```bash
   docker build --no-cache -t mediapipe-build .
   ```
3. To test the bazel build inside the container at par with how you would run it outside of a docker container:
   ```bash
   docker run --rm -it -v "$PWD":/mediapipe mediapipe-build /bin/bash -c
   ```
   then inside the running container's interactive prompt:
   ```bash
   "cd /mediapipe && bazel build -c opt --define MEDIAPIPE_DISABLE_GPU=1 --copt=-I/usr/include/opencv4 mediapipe/python/solutions:hands"
   ```

### 3. Build and Use the MediaPipe Python Solution (Meaning, Stabilize that last step) 

When we want to change the mediapipe original C++ pipeline such as for developing a non-mediapipe-framework port of it in C++, we only care about building mediapipe as above. But if we want to use the python solution for verification of its building python bindings, we need to also stabilize the last step of building and installing a python package ― the repository has its original implementation for this build, but it needs to be restabilized as per the reasoning of the beginning of this document about why that stability drifted away since v0.10.13 was released.

**Modern pip restrictions**: You will have to address the consequence of stricter dependency handling in newer pip versions, which were not an issue at the time of v0.10.13's release.

Building MediaPipe using the legacy `setup.py bdist_wheel` mechanism is deprecated and will be removed in a future version of pip. To future-proof your build process, use the standardized build interface by running:

    pip install . --use-pep517

see [pip issue #6334](https://github.com/pypa/pip/issues/6334).

#### Last Pip Issue Encountered

Having pip reuse the already Bazel built OpenCV instead of trying to build OpenCV from source from scratch, is a good idea. However:
To build and install the MediaPipe Python package using the already bazel built OpenCV (recommended for faster builds):

```bash
MEDIAPIPE_LINK_OPENCV=1 pip install . --use-pep517
```

- This will instruct the build to use your system OpenCV and skip building OpenCV from source.
- Make sure OpenCV and its development headers are installed in your environment.
- Do **not** use `--install-option` with pip, as it is not supported with modern builds.

However its some work to get to that working.
  
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

## Run the Python Example (Inside Container)

To verify the built mediapipe with a video file (place `input.avi` in `mediapipe/python/`):

```bash
python3 mediapipe/python/verify.py
```

- Check `mediapipe/python/verified-detections.json` for results after running `verify.py`.

# Python Package Versioning

The Python package version is now automatically set at build time to `0.10.13+git.{commit_hash}` (where `{commit_hash}` is the current short git commit hash). This ensures every build is uniquely versioned and PEP 440 compliant. You can use AI to reproduce this feature, and add whether git is dirty or not to the version string. As much as versioning it is relevant for the use case.

# Additional Notes

- The Docker image can build all MediaPipe targets; adjust Bazel build targets as needed.
- All patching and setup steps should be ultimately captured by a working Dockerfile for short-term reproducibility (up until the reasons for no 100% stability work for a long enough time again out there).

