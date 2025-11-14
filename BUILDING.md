# About this Branch

This branch has accomplishes a safe and parity-verified reduced mediapipe v0.10.13 hands pipeline, namely it reproduces its work with performance no lesser, having been safely and gradually having been reduced to:
1. a no-pipeline single-component implementation of the same as the original.
2. a single-calculator mediapipe pipeline, which merely wraps the former component.  

This work was made possible by developing a set of verification mains which verified after each step of the gradual liberation process, that the output per frame of the current code is the same as that of the original pipeline:
+ a C++ main comparing the current code's output per frame to output yielded from the original v0.10.13 tag code's run over the same input video, through branch `buildable-reference`.
+ `buildable-reference` still includes that main which produces an output file when running the original pipeline over a given input video, to enable these parity testing workflows mentioned, for repeating the same parity testing process over any new video file. 
+ a python main performing the same verification.
+ a C++ main utilizing newly made C API bindings for the same component, which is also performing the same verification.
+ a rust main which does the same, using rust FFI bindings made for the said C API facade wrapping the C++ component. 

1. These quickly ascertained after each pipeline reduction step, that the pipeline still produces the same output as before, thus safely facilitating the liberation development process. They can be reused for verifying other pipelines' liberation work if applicable (e.g. face landmarks tracking).
2. They can be repurposed or evolved to juxtapose results rather than strictly compare output for equality. 

GPU support was not tested and not across all the many disparate code locations providing GPU support in mediapipe, in this branch, preserved.<br>
That is because GPU support in this pipeline was originally scattered across many layers of the mediapipe stack: both many of its framework layers, and within many calculators themselves. The GPU supporting code elements have been preserved in this branch only on a best-effort basis and not where they significatnly hindered clarity and code surface; GPU support when applicable, should be better developed from scratch, with a lot of attention to detail. 

# Development Environment and Build Guide

## Build and Development Environment Setup

Successful building on Ubuntu 24.04 as well as building a Docker image (Ubuntu 22.04 based) that can build mediapipe at this revision have been accomplished, and reproducible as per below, as the basis for reducing the hand processing pipeline from mediapipe into liberation from use of mediapipe framework.

Currently successfully building with Bazel 6.5.0 as also seen in the [Dockerfile](Dockerfile).

Unlike original mediapipe, this build builds a specific version of OpenCV from source which avoids [protobuf runtime issues](https://github.com/nui-ai/mediapipe/issues/18). This is also more responsible in the general case, as there is no reason to get an arbitrary version and build of OpenCV just because it is globally installed on the system.

### Jetbrains as IDE
1. Installing the Bazel plugin for proper IDE operation in JetBrains IDEs:<br>
    + a _must-have_ plugin if you use Clion as your C++ IDE, as it makes the IDE understand the project strutcture by running bazel info commands on its sync, which is super-essential for code navigation between bazel BUILD files and their respective C++ source files and vice versa by your normal "go to definition" keyboard shortcut and through its context option "go to corresponding BUILD file" on the context menus. without this plugin, most of the source code will be flagged red as unknown symbols, and you won't have other conveniences like run configurations for building and running through bazel and you basically aren't getting any proper IDE experience at all!
    + https://www.youtube.com/watch?v=GV_KwWK3Qy8
    + https://ij.bazel.build/docs/bazel-plugin.html
    + Why this plugin?
      + Chiefly, this plugin make the editor aware of `pb.h` header files which are only generated during each bazel build, outside the codebase source-tree, from the underlying protobuf definitions which mediapipe uses for mostly all of its C++ classes; these `pb.h` are expected by C++ include statements, without which most code symbols are marked as unknown in the editor, rendering code editing noisy and unusable.
      + The editing awareness through these plugins is only materialized after you trigger the "bazel Sync" action of the bazel plugin, from the Bazel menu, the IDE icon, or context menus ― it basically draws the bazel build graph into the IDE's project model to make the IDE understand the project code. You have to repeat this plugin sync after build file changes. this sync is not really related to the bazel sync commmand so
        much, for the fine relationship of the plugin's sync operation to the bazel sync command see here ― https://chatgpt.com/s/t_6903a8d2adf08191ba71a9d65f4510c8.
      + They also enable bazel build and run configurations which you use for building, running, debugging ...
      + Case in point: jumping from a CC source file directly to its very build definition in the respective BAZEL file and vice versa! (via left and right click options)
<br><br>
   + 👉 **Clion**: install the JetBrains plugin called `Bazel for Clion` (in Clion)
<br><br>
   + 👉 **PyCharm**: install the JetBrains `Bazel` for the equivalent support in PyCharm, but using Clion is more relevant most of the time.
<br><br>   

2. 👉 Install the `Protocol Buffers` plugin from Jetbrains (in Both IDE).
3. 👉 Only optionally add the `FlatBuffers` plugin from a 3rd party developer.
4. Do not install conflicting Clion plugins for Protocol Buffers IDE support, they will make the IDE silently fail on many features and become defunct, which is a known issue in the JetBrains plugin framework not marking them as conflicting plugins. Just use the JetBrains one.

5. 👉 [Set-up preqrequisites for rust](rust/README.md).
 
 
+ Note: as a byproduct of the Bazel for Clion plugin, Clion uses the `.clwb` (Clion With Bazel Acronym) path as the working directory in many contexts; you usually want to revert to the project root path one level back upwards, for running stuff.

### VS Code as IDE
  + VSCode has popular bazel extensions which provide a different set of features; some things from those VSCode plugins you'd want in JetBrains and vice versa.
  + Let CoPilot port the project's JetBrains run configurations into VSCode's equivalent json files for the same.
  + VSCode is generally more configurable and extensible for you to code your way to make the extensions do more. 

### Bring Your Own Test Video
place an avi video file with hands motion in the project root path, to enable the test-running the pipeline on it:
  + typically the video files are too large for git/github, so bring your own by using Gesture Studio.
  + you'd ultimately provide its name to any run command which uses our hand tracking from among the repo's built runnables.
  + the builds per-se do not require this video file to be present, it is only used at runtime as hand tracking input.
  + you can use [Gesture Studio](https://github.com/nui-ai/core?tab=readme-ov-file#gesture-studio) to generate your own video, give or take using ffmpeg to convert to avi if needed. 

# Initial Local Setup
0. clone this repository and cd into it.
1. [.bazelproject](.clwb/.bazelproject) controls which parts of the huge mediapipe codebase are built by bazel, and which files should be ignored by the bazel plugin to avoid over-indexing. 
    + review it, make sure you understand it, and update it wisely when needed. 
    + you can use the git version of it, but this requires active action in the case of JetBrains IDE:<br>
      the plugin generates it from scratch when you initally import the project, so you can only manually copy its git versioned copy into there after it has been generated automatically by the plugin on project initialization when first opening the repo as a bazel project in a bazel plugin enabled IDE.
2. replace all run configurations in your JetBrains IDE to use your own bazel cache path of choice instead of /home/matan/.cache/bazel-disk-cache. some of the git included run configurations build the targets, some run them. update all of them bazel run configs to use your own local bazel cache path, by a "find and replace" of their underlying xml files under [.run](.run). 

# Build Verification Flow ― python oriented only

verify that the C++ build is fine:
   ```bash 
         bazel build \
         -c opt \
         --define MEDIAPIPE_DISABLE_GPU=1 \
         --define OPENCV=source \
         --disk_cache=/home/matan/.cache/bazel-disk-cache \
         --fission=no \
         //mediapipe/examples/desktop/hand_tracking:hand_tracking_pipeline_run \
         //mediapipe/examples/desktop/hand_tracking:libhands_pipeline_operator_c_api.so \
         //mediapipe/examples/desktop/hand_tracking:hand_tracking_pipeline_run_via_c_api \
         //mediapipe/examples/desktop/hand_tracking:compare_pipelines_output_data \
         //mediapipe/examples/desktop/hand_tracking:hand_tracking_no_pipeline_run \
         //mediapipe/python:_framework_bindings
   ```
   see here https://chatgpt.com/c/690f1afd-57cc-8327-8ce3-5ce9ca9e2713 for memory safety warnings in google dependency packages which are expected.

1. make and activate a python 3.12 venv, or start a terminal in your PyCharm project view of this repo. 
    ```bash
    python3.12 -m venv .venv
    source .venv/bin/activate
    ```
2. run the below `pip` command to both build a python package providing python api to the mediapipe C++ core as a python package, and to install that package to the currently active python environment. <br>
   + [setup.py](setup.py), which is triggered to run by the below `pip` command, runs with PEP 517 isolation,
   + it invokes bazel to build all required C++ bazel build targets before building the resulting python package.
   + it may or may not reuse the bazel cache from the previous step (need to double-check that) despite its pip build isolation. 
   + it builds all required C++ targets, plus the python bindings and cumbersome fiddles which [setup.py](setup.py) does for building the mediapipe python package and installing it to the current python environment. it is able to install it to the active python venv thanks to exporting the below environment variable prior to issuing the pip command. without the export preceding the below `pip install` command, mediapipe will fail to import in the active python environment.
   + to reiterate: without the preceding export of the python environment variable, you are installing the built package to an arbitrary location (which may in edge cases also degrade the bazel building outside of the python use case). 
   + to build and install the python package, run after activating your project's python venv:
       ```bash
       MEDIAPIPE_PYTHON_BIN=$(which python) pip install . -v
       ```
   + without `-v` stdout is swallowed unless pip fails entirely, but its stdout is useful when wishing to follow what it is doing.   

3. verbose bazel analysis logs created when running under this pip command become available at `/tmp/bazel.explain`, they explain some of bazel's caching decisions.

5. run the python verification main which should complete with exit code 0, it doesn't show any fancy results but only records the pipeline's output per input to a protobuf file:
    ```bash
    python3 -P hand_tracking_pipeline_run.py
    ```

This concludes building and running the hand tracking from python.

# Build Verification Flow ― Full  

**Verify all of the many intended buildables of this repository:**
<br>
1. build all bazel build targets, using the repo included JetBrains run configuration from `.run/build all liberation project targets.run.xml`, or by mimicking it as below:
and mimicking their build command:
    ```bash
    bazel build \
            -c \
            opt \
            --define MEDIAPIPE_DISABLE_GPU=1 \
            --define OPENCV=source \
            --disk_cache=/home/matan/.cache/bazel-disk-cache \
            --fission=no \
            //mediapipe/examples/desktop/hand_tracking:hand_tracking_pipeline_run \
            //mediapipe/examples/desktop/hand_tracking:libhands_pipeline_operator_c_api.so \
            //mediapipe/examples/desktop/hand_tracking:hand_tracking_pipeline_run_via_c_api \
            //mediapipe/examples/desktop/hand_tracking:compare_pipelines_output_data \
            //mediapipe/examples/desktop/hand_tracking:hand_tracking_no_pipeline_run \
            //mediapipe/python:_framework_bindings 
    ```        
    + each of the above bazel targets yields an executable or a shared library as per its name, the above command runs them all.  
    + Then, run the built C++ mains through their respective JetBrains run configurations from `.run/`, or by mimicking their bazel run commands, from under the project root path.
   <br>
2. build and run the rust main/s which are using the C++ based hand tracking and included in this repo, after you have built the above bazel targets: 
   + build:
       ```bash
       cd rust
       cargo build
       ```
   + run the rust mains through the provided rust run configuration, or by mimicking their `cargo run` command, from under the `rust/` subdirectory, e.g.:
     ```bash
     cargo run -- --graph-file mediapipe/modules/hand_landmark/hand_tracking_pipeline.pbtxt --input-video-path ../sample-video-two-hands.avi
     ```

3. pip install to generate and install the python package built by this repo into your active python venv, as per above, and run the main python code which uses it as per abvoe too.

## Build Cleaning

Only for those rare cases where you don't trust any of the build systems' incremental awareness, and shouldn't typically be needed.

+ to clear all bazel build caching:
    ```bash
    bazel clean --expunge && trash /tmp/bazel-\$\{USER\}/ && trash ~/.cache/bazel/
    trash /home/matan/.cache/bazel-disk-cache
    ```
   + This should be stressed: a mere bazel clean --expunge is not enough to clear _all_ bazel caches. See the end parts of https://chatgpt.com/c/68ce82f1-d284-8327-90a0-e4980994cf35 for a delination of what it clears. the above trashing of specific paths is aligned to the way that this repository uses specific caching paths after we added a fixed cache location for it to avoid it from avoiding incremental building by pip's ephemeral isolated build environments.
   + This does not clear the wheel installed binary of mediapipe which `pip install .` installs into the active python environment! only `pip uninstall mediapipe` does that!
   + This does not uninstall the mediapipe python package (only a `pip uninstall mediapipe` does;  `pip uninstall` caused the known issue described below, but after recent changes no longer does).
   + Sometimes manually trashing the built executable is also necessary (at least, encountered only during some earlier stages of development). 

+ to clear the cargo build which builds the rust code of this project which runs our pipeline and our no-pipeline derivation:
    ```bash
    cargo clean
    ```
  
+ to clear the pip installed mediapipe package built by the setup.py of this repository from any target venv to which it was installed:
    ```bash
    cd <path to the target python venv> ...
    pip uninstall mediapipe
    ```

## Debugging

For being able to place breakpoints in the bazel built C++, it is necessary to use the following build flags in the bazel build commands in order to build the top-level build targets (executable, lib) as really debuggable:
+ `-c dbg` (or any equivalent which generates debug symbols)
+ `--fission=no` (this option simplifies the debug symbol's writing into the executables, thus enabling Clion/GDB/LLDB to use them; without it, you cannot place breakpoints in the C++ code and expect them to actually work).
+ the same bazel flags should be used on the run configuration as in the build run configuration, otherwise the run configuration will trigger a rebuild with the default bazel build options.

Running the built executables not directly but rather through bazel (by using a `bazel run` command, or the IDE's bazel plugin provided run configuration types) rather than directly running the executable, may be required for being able to set breakpoints. Running through the bazel command also makes ABSL_LOG messages more ergonomic (a hyperlink to the source code line where they are issued from is attached to each one of them in JetBrains IDE).

## Deploying to Other Environments
Unlike rust which just needs to find a lib built by C++ build, python can only use the built wheel package which `pip install .` installs into the active python environment.<br>
When you work within the current repo, this is what pip installing does.
<br><br>
There's a pethora of ways to get the mediapipe python package built by this repo, into any other project environment, instead of stipulating one way, the full gamut of options is discussed here: https://chatgpt.com/s/t_6917132141d48191abdf9d9eb1e4217b.
In general there are two ways to get the mediapipe python package built by this repo into any other project environment:

1. reuse this repo's `pip install` process to rebuild and install the wheel it's building into your target other project environment.
2. install the python wheel built by this repo into any other python environment like any local wheel installation, namely just install last wheel that `pip install` (the exact pip install command from above) has already built in this repo, into your target python environment.

+ recall that our `pip install` command both re-builds the mediapipe python package and installs it into a target python environment.

## see also
https://github.com/nui-ai/mediapipe/issues/18

---

### ⚠️ Obsolete: Build Known Issue ⚠️  
Sometimes you get this error from python, or a similar one from C++ mains:
```
Failed to load resource: mediapipe/modules/palm_detection/palm_detection_full.tflite
```
Probably only after doing a `pip uninstall mediapipe`, which seems to reproduce this behavior.

**Solution:** rerun the pure bazel build, after repeating the `pip install` for the python environment's sake. somehow this places something where it needs to be to avoid this error happening.  

Conclusions: 
1. Something must be not perfect enough still, if this can happen.
2. The pip installed mediapipe is not fully isolated from the (non-pip) Bazel-built mediapipe afterall, even though pip mostly uses build isolation for its bazel building.  

**How to solve it from happening:**<br>
The tflite model file actually used in the normal non-fail scenario is named `hand_landmark_full.tflite`. It contains both that palm detection model and the landmarks model, and when it's not found, the runtime tries looking for the palm model in default locations where it shouldn't be, and issues that error. This file is available both under the bazel `./build` and under the virtual environment directories alike, but it's not a symlink from one to the other.  

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
bazel build -c opt --define MEDIAPIPE_DISABLE_GPU=1 mediapipe/python/solutions:hands
```

- This command builds only the hand landmarks solution for Python, matching the Docker image build step.

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
