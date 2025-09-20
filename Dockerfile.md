## The Ubuntu 24.04 Docker Image

+ This repository's Ubuntu 24.04-based [Dockerfile](Dockerfile) has two roles:
  1. Reproducing the build of mediapipe v.0.10.13 in a clean container ― this shows reproducibility in building mediapipe as no host system dependencies/configurations are involved.
  2. It was also adapted for use as a GitHub Actions self-hosted runner, but that part of it is commented now, as it failed on the github side without proper error messages.
 
+ The included [Dockerfile.manylinux_2_28_x86_64](Dockerfile.manylinux_2_28_x86_64) 
  + Has the same objective, and if successfully builds and is able to successfully run the mediapipe build, then its build artefacts may be used in a manylinux_2_28_x86_64-compliant Python wheel which can basically run on any modern Linux distribution. This was never tested by me, but I did update it to use the system OpenCV which was one modification required for building mediapipe on Ubuntu 24.04. 
  + Maybe its building for any linux can be integrated into the now verified Ubuntu 24.04 Dockerfile.
  + Not sure whether that Dockerfile gets all the header files required for the build, maybe it simply doesn't. 

*The GitHub Actions runner binary (now commented out in the Dockerfile) downloaded and extracted in the image, but its github cloud-side registration should be performed at runtime. The version of this runner being downloaded should become obsolete quite fast.

