# Update About Successful building on Ubuntu 24.04 as well as building a Docker image (Ubuntu 22.04 based) that can build mediapipe as well.

The current commit reflects the exact code revision of git tag v0.10.13 of the original MediaPipe repository, with some patches applied to make it buildable again, and it now successfully builds the hand tracking pipeline target as well as the python package that exposes the mediapipe api to python as a python package called `mediapipe`, installable into a python environment by `pip install .` as per the instructions below."

# Build and Development Environment Setup

+ Currently successfully building with Bazel 6.5.0 as also seen in the [Dockerfile](Dockerfile).
+ Consider installing the following JetBrains Plugins for IDE support:
  1. `Bazel for Clion` (in Clion) is a must-have plugin if you use Clion as your C++ IDE, as it makes the IDE aware of the bazel build graph and its generated files, which is essential for code navigation and understanding in a bazel-built project like this one.
  2. `Bazel` for the equivalent support (in PyCharm).
  2. `Protocol Buffers` from Jetbrains (in Both IDE).
  3. Only optionally add the `FlatBuffers` from a 3rd party developer.
+ Why plugins?
  + Chiefly, the first two plugins make the editor aware of `pb.h` header files which are only generated during each bazel build, outside the codebase source-tree, from the underlying protobuf definitions which mediapipe uses for mostly all of its C++ classes; these `pb.h` are expected by C++ include statements, without which most code symbols are marked as unknown in the editor, rendering code editing noisy and unusable.
  + The editing awareness through these plugins is only materialized after you perform the "Sync" action in the Bazel menu or in context menus ― it basically draws the bazel build graph into the IDE's project model to make the IDE understand the project code.
  + They also enable more fluency with bazel run configurations and stuff in the IDE.
  + Don't install conflicting Clion plugins for Protocol Buffers IDE support, they will make the IDE silently fail on many features and become defunct. Just use the JetBrains one.
+ As a byproduct of the Bazel for Clion plugin, Clion may use the `.clwb` (Clion With Bazel Acronym) path as the working directory or the console home directory; you usually want to revert to the project root path for running stuff.

# Building 

0. clone this repository and cd into it.

1. verify the building of some mediapipe C++ bazel build targets which the python build will rely on, as you might need to rebuild them after the pip install as per the known issue documented below.:
    ```bash
    bazel build -c opt --copt=-I/usr/include/opencv4 --define MEDIAPIPE_DISABLE_GPU=1 \
    //mediapipe/examples/desktop/hand_tracking:hand_tracking_tflite \
    //mediapipe/examples/desktop/hand_tracking:hand_tracking_cpu
    ```

8. above we dealt with one example C++ main which runs the hands pipeline. but there are three C++ example mains which build and use the hand tracking pipeline, all of which run it over an input video. 
   + their build commands and run commands are originally described in [their shared build file](mediapipe/examples/desktop/hand_tracking/BUILD)
   + they are named very non-descriptively:
   + `hand_tracking_cpu`: applies the hands pipeline to an input video file, and writes a video file with the pipeline's results drawn on it. this is the one we used above.
   + `hand_tracking_tflite`: applies the hands pipeline to an input video file, and writes a video file with the pipeline's results drawn on it. and shows each image with its results drawn on it, in an OpenCV window as it goes.
   + `hand_tracking_gpu`: applies the hands pipeline to an input video file, and writes a video file with the pipeline's results drawn on it. and shows each image with its results drawn on it, in an OpenCV window as it goes. Uses GPU for inference.
   + the naming could not be more awkwardly non-descriptive, but now you have the mapping for what each does.
   + all defined as said in the same build file. 
   + if you want to only build the hands pipeline (yes, a mediapipe graph should be built, not only fed as a pbtxt file to C++ code running it) you can build the bottom line target of the three, which you find in the `deps` field of their build definitions in the said build file.
   + most of the above C++ mains are used by the python api, they are mostly other demo runners from the original mediapipe codebase.

1. make and activate a python 3.12 venv:
    ```bash
    python3.12 -m venv .venv
    source .venv/bin/activate
    ```
2. run the python build, which triggers bazel to build the hand tracking pipelines and underlying mediapipe framework before building and installing the python wheel which provides the python mediapipe api. the included `setup.py`, triggered to run by the below `pip` command runs bazel under the hood to build all C++ dependencies required for the hands model. this not only builds all required C++ targets, but also the python bindings and cumbersome fiddles that `setup.py` does for building the mediapipe python package and installing it to the current python environment. Note that without the preceding export of the python environment variable, `pip` will cause bazel to rebuild from scratch for any source change when used by pip (as a direct consequence of modern pip's build isolation feature). without the export, mediapipe will also fail to run from python. so you want that export command before you use pip here:  
    ```bash
    export MEDIAPIPE_PYTHON_BIN=$(which python)
    pip install . -v 
    ```
    `-v` should be used as otherwise due to pip's build isolation stdout is swallowed when the build does not fail altogether, making its steps hard to trace if needed.
4. verbose bazel analysis logs created when running under this pip command become available at `/tmp/bazel.explain`, they explain some of bazel's caching decisions.

5. place a video file with hands in the project root path, as `sample-video.avi`, and run both of the following tests:<br>
   + typically the video files are too large for git/github, so bring your own by using Gesture Studio. 
   + C++ program built by the bazel build, this program takes in the sample video, and produces a derivative of it which shows the pipeline's predictions (and bounding boxes detected internally by it) per frame of the original video:
       ```bash
        bazel-bin/mediapipe/examples/desktop/hand_tracking/hand_tracking_tflite --calculator_graph_config_file=./mediapipe/graphs/hand_tracking/hand_tracking_desktop.pbtxt --input_side_packets=input_video_path=sample-video.avi,output_video_path=output_video.mp4 
       ```
     this verifies the C++ built hands pipeline without relying on any python-targeting parts of the build.<br>
   + python test which should complete with exit code 0, it doesn't show any fancy results but record the pipeline's per input output to a protobuf file.
        ```bash
        python3 -P hand_tracking_pipeline_run.py
        ``` 
        note that without `-P` python will try loading python modules from the `mediapipe` directory under the project's tree root, which is essentially our python "source directory", which is a horrible entanglement, and is also bound to fail since some of the mediapipe python modules are only dynamically built & placed (into the active python environment) by the pip install process ― because most of mediapipe python-exposed sub-packages are either pybind generated or bazel generated from protobuf definitions or both (e.g. mediapipe.python._framework_bindings). So the python source tree _never_ contains all the modules that the mediapipe python api expects to find, but it can sure throw you off track with cryptic module loading errors if you try to run without -P and thus let python first look for modules under the python source directory `mediapipe`. With this project you only want python to run from the active python environment, not from its "source" path, which is what `-P` does.
   
6. **to install the same built package into another target python environments:** the former pip command only built the mediapipe python package (which itself uses the necessary C++ bazel built lib files) and installed it into the active python environment (the venv you created and activated above). **to install the same built package into another target python environment** for example one associated only to a downstream project's venv, use the following steps:
    + use pip to rebuild in a way not throwing away the wheel file, but keeping it under the `dist/` directory:
    ```bash
    python -m pip wheel . -w dist 
    ```
    + this leaves the wheel file under `dist/`, so after this you can run `pip install ...dist/mediapipe-0.10.13.dev0-py3-none-any.whl` in any target python environment to install the built mediapipe package from here into that environment (if the package .whl file name changes, replace with its name as given by the pip install output). the `...` portion is only a placehoder to substitute: you should replace the relative path to the wheel file in the command just given, to one that points to the wheel file from the target environment's perspective, before running that command. you typically want to `pip uninstall mediapipe`in the target python environment before this step.  

# ⚠️ Build Known Issue in this branch ⚠️  
You get this error from python, or a similar one from C++ mains. this only happens on this branch and solved by radical changes in the liberation branch:
```
Failed to load resource: mediapipe/modules/palm_detection/palm_detection_full.tflite
```
This is caused by a `pip uninstall mediapipe` or after re-running the `pip install .` after the first run of it.

**Solution:** if you get that message when trying to run hand_tracking_pipeline_run.py, rerun the pure bazel build, using the command from the previous section. that places those files which the python build probably only links to. after this you can run `hand_tracking_pipeline_run.py` and you won't get that error. this is fixed in the liberation branch but hard to trace and backport here, so just go through this extra step in this branch when you hit that.  

Conclusions: 
1. Something must be not perfect enough still, if this can happen.
2. The pip installed mediapipe is not fully isolated from the (non-pip) Bazel-built mediapipe afterall, even though pip mostly uses build isolation for its bazel building.  

**How to solve it from happening:**<br>
+ The tflite model file actually used in the normal non-fail scenario is named `hand_landmark_full.tflite`. It contains both that palm detection model and the landmarks model, and when it's not found, the runtime tries looking for the palm model in default locations where it shouldn't be, and issues that error. This file is available both under the bazel `./build` and under the virtual environment directories alike, but it's not a symlink from one to the other.
+ Update: this is solved in the liberation branch, but in the current branch you still need to workaround this issue by re-running the bazel build after the `pip install` command from above.

```
For a Python package installed via pip, the list of installed files is recorded in the package metadata, typically in a file named RECORD (for wheels) or installed-files.txt (for legacy setup.py install). These files are located in the package's .dist-info or .egg-info directory inside your Python environment's site-packages.
For your mediapipe package, after installation, you will find a directory like mediapipe-0.10.13.dev0.dist-info in site-packages. Inside, the RECORD file lists all files installed by the package, including:
All Python modules and packages (e.g., mediapipe/__init__.py, other .py files)
Compiled extensions (e.g., .so files built by Bazel)
Data files included via include_package_data=True
Any other files specified in MANIFEST.in or by setuptools
You can inspect the RECORD file directly to see the full list of installed files for your package.
```

`pip uninstall` removes `hand_landmark_full.tflite` file from the python venv as part of its action, but it doesn't remove the `./build` copy of it, so something deeper is preventing both non-python and python mediapipe from using that file in this fail scenario, which stems from the `pip uninstall` run (maybe it is leaking some action outside of the python environment domain).<br>
This can be ironed out as part of the wider https://github.com/nui-ai/mediapipe/issues/18 where it is more important to avoid this file being a downloadable than fix this minor bump which has its full workaround above 🎯

🛈 Notes:
1. Reproducibility:
   - The included Ubuntu 24.04-based [Dockerfile](Dockerfile) was created and tested to contain the OS-level dependencies needed for a successful mediapipe v0.10.13 build, and fully tested to reproduce a successful build, so this process is reproducible by this Dockerfile and not an artefact of special conditions on my machine ― the built docker image fully reproduces the error-less build of mediapipe at its v0.10.13 commit level, however for even more future proofing:
   - A todo item is to upload that built docker image to future-proof it from reliance on Internet repositories of dependencies which may change or disappear in the future. 
   - [The other included docker files](Dockerfile.md), provided originally by mediapipe's original codebase, were not tested.
2. The changes having been made for current-day buildability are documented in the git commits trail. 
3. Maybe `pip install` builds a bit more than we need as we didn't modify `setup.py` to only build only the hands target as `bazel build --config=cpu-only -c opt //mediapipe/examples/desktop/hand_tracking:hand_tracking_cpu` would, though most of the bazel build time is the shared mediapipe framework anyway. 

# bazel clean

the bazel cache might be used across different branches of this repository, whether you have each branch in its own local project copy, or not. 

to explicitly clear bazel build caching:

```bash
bazel clean --expunge && trash /tmp/bazel-\$\{USER\}/ && trash ~/.cache/bazel/
```

+ This should be stressed: a mere bazel clean --expunge is not enough to clear _all_ bazel caches. See the end parts of https://chatgpt.com/c/68ce82f1-d284-8327-90a0-e4980994cf35 for a delination of what it clears. the above trashing of specific paths is aligned to the way that this repository uses specific caching paths after we added a fixed cache location for it to avoid it from avoiding incremental building by pip's ephemeral isolated build environments. 
+ This does not clear the wheel installed binary of mediapipe which `pip install .` installs into the active python environment! only `pip uninstall mediapipe` does that!
+ This does not uninstall the mediapipe python package (only a `pip uninstall mediapipe` does, but don't do that, just repeat the pip install for updating, as `pip uninstall` causes the known issue described below).

# old deprecated comments (but has some gems in it to be extracted into above)

This guide explains how to work towards reproducibly building MediaPipe v0.10.13 for just the hand tracking target, at revision tag v0.10.13 of mediapipe which this forked repository was reverted to.
Judging from experience you need to work a few days to make it happen, as the build code will fail with modern Bazel, versions of dependencies it will fetch from the Internet which are not the same as when this build was originally working at the time of v0.10.13 release, and similar issues with its last-mile pip install for python proof of concept. Sometimes AI gets it right after just one day of careful iteration.

This repository is also a codebase where a lot of voodoo took place on a first two-day run of this all ― but it contains the right build target commands named below, and a lean verification script `verify.py` that can be used to verify the build result when you get that far.

Techically it's a fork of the original MediaPipe repository, reverted to the v0.10.13 tag commit, with some patches applied to make it buildable again which didn't go all the way and should be restarted from v0.10.13 from scratch.

This repository would be between a prerequisite and a starting point for deriving a mediapipe-framework liberated C++ implementation of the hand tracking pipeline, which is the topic of the sibling repository [hand-tracking-cpp](https://github.com/nui-ai/mediapipe-liberation)

- Success means:
  - A build command that's more specific than the original MediaPipe build instructions to avoid build time or even errors from unnecessary build targets which the hand tracking pipeline does not require.
  - Building a docker image to have the necessary OS dependencies for reproducibly running the build independently of the host system environment
  - Actually running the build inside a docker container using that docker image
  - Running the included Python verification script inside the container as well (if desired)

# Why this is never 100% future proof (outside the case of a docker image)
- The build process relies recursively on dependencies being fetched from over the internet. These dependencies may change, be removed, or otherwise become incompatible with the build process. Which is why we needed the changes comited on this forked repository, and why other changes may arise as necessary in the future. 
- We can pin down specific versions all across and hope they keep served on the Internet for long enough, to change the tradeoff a little.
- In a docker image built as per the docker image build command included below, all these dependencies become baked into the image and thus future proof one level more deeply ― as long as we can run that built image they are baked in, regardless their availability on the Internet or how newer versions of pip or bazel behave differently against the Internet repositories of dependencies where they come from.

# Why this is hard
Its a tug of war between old versions that can no longer install or work against the Internet repositories as they are today, and new versions that don't like elements of how the build (which was coded to old versions) is. this statement applies to both the Bazel and the pip parts in equal amounts! See also https://github.com/copilot/c/090b58a9-b989-4d7c-afaa-a7e0f5131239 for why it is hard (unstable tflite and tflite dependencies ecosystem in recent years).

# Guidelines for Stabilizing the Build Process

## Build Stabilization Recipe

To stabilize the MediaPipe build process, follow this step-by-step approach:

### 1. Stabilize the build on your Local Ubuntu Machine ― Done.

Stabilizing it without docker means faster turn-over times. Done. You may choose to skip this step if you prefer to go directly to the Docker environment. Maybe that's a good idea. 

### 2. Stabilize a Docker image that succeeds in building the target ― Done.

Once a local build works, or if you think your local machine is dirty or just prefer to skip it:

A Dockerfile already exists in this repository, which you can modify as needed or start from scratch.

1. Different versions of Ubuntu can play out differently in which versions of what they agree to install ― meaning different complexities, different bugs.
2. To build the Docker image:
   ```bash
   docker build --no-cache -t mediapipe-build .
   ```
3. To test the bazel build inside a container started from that image:
   ```bash
   docker run --rm -it -v "$PWD":/mediapipe mediapipe-build /bin/bash
   ```
   then inside the running container's interactive prompt:
   ```bash
   git config --global --add safe.directory /mediapipe
   pip install .
   ```
5. you only need `pip install .`, which builds all necessary mediapipe targets as per the `setup.py` instructions. if it worked, you're done. no need to run a bazel build yourself.

6. The resulting docker image is tagged as `mediapipe-build` and stored in your local machine's Docker image registry. The above does not push the image to any remote repository; it only exists on your local system unless you explicitly push it elsewhere, unless we uploaded it to e.g. serve from github's ghcr.io or dockerhub. Rebuilding it from the current repository takes only a few minutes, but having an image on the cloud can give more assurance because it does not rely on Internet servers being available to serve all OS, bazel and pip dependencies which it needs to fetch, which are already baked into a successfully built image. Actually, the image now prebuilds mediapipe as part of its Dockerfile, so that all Internet dependencies are baked into the image, and then rebuilding with only code changes does not need to fetch anything from the Internet ― this can make it stand the test of time as the Internet repositories of dependencies phase out old versions of dependencies. 

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
  
## CPU-Only Hand Tracking Build Command

This PR implements a minimal CPU-only build configuration. Use this command to build the hand tracking target:

```bash
# Set environment variable to bypass blocked releases.bazel.build
export BAZELISK_BASE_URL=https://github.com/bazelbuild/bazel/releases/download

# Build the CPU-only hand tracking target
bazel build --config=cpu-only -c opt //mediapipe/examples/desktop/hand_tracking:hand_tracking_cpu
```

**Note**: The build will fail due to SSL certificate verification issues when Bazel downloads dependencies from GitHub in CI environments. This is a confirmed persistent issue. Solutions include:

**For CI/Production Environments:**
- Use pre-downloaded dependencies approach (recommended)
- Configure Docker environment with proper Java certificate store
- Use dependency caching to avoid repeated downloads

**For Local Development:**
- Local Ubuntu environments typically work with proper SSL setup
- Ensure `ca-certificates` and `ca-certificates-java` packages are installed
- Configure Java to trust system certificates: `sudo update-ca-certificates`

**Workaround for Testing:**
The SSL issue specifically affects Bazel's Java-based download mechanism. Manual downloads work fine:
```bash
# This works:
wget https://github.com/bazelbuild/bazel-skylib/releases/download/1.3.0/bazel-skylib-1.3.0.tar.gz

# This fails in CI:
bazel build --config=cpu-only -c opt //mediapipe/examples/desktop/hand_tracking:hand_tracking_cpu
```

# Why this Recipe Matters

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
