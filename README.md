# Mediapipe Hand Tracking Radical Improvement Track

## Current Branch Objective
+ This branch is mediapipe v0.10.13 updated to be present-day buildable. Through lots of labor it seamlessly builds the mediapipe code equivalent to its v0.10.13 code revision.
+ It is enriched with humble python scripts which parse the main pipeline (which is rather recursive) into human and machine readable formats which help with navigation of its involved structure.
+ The included JetBrains run configurations help with building and testing the mediapipe code.
+ It should not be used for development of modifications to mediapipe of that version nor for developing derivatives thereof, but be kept as a reference and proof of build.

## Objective
The objective of this repository is detailed in https://github.com/nui-ai/mediapipe-liberation/blob/main/README.md, which is gradually being ported into the current repository.

## Why v0.10.13?
+ Mediapipe is not the kind of project you'd want to absorb new versions of in terms of its project governance.
+ Later versions migrated to be (further) enveloped by what they call "MediaPipe Tasks", which [had breaking api changes, didn't work as well for our use cases, and brought no benefit](https://github.com/nui-ai/core/blob/4c09af2dd4a10df2b83c1ac3d2855182756cf270/nui/mediapipe/mediapipe_tasks_api/Mediapipe.py#L1-L20).
+ Later versions shifted focus away from python use, demonstrating little to no care for solving python api issues, so regressions can only be expected.
+ The underlying neural network hand detection and hand landmark models were never updated in later versions (and may never be upgraded at all).
+ Versions much earlier than v0.10.13 had noticeably inferior hand tracking accuracy.
+ The [v0.10.13](https://github.com/google-ai-edge/mediapipe/releases/tag/v0.10.13) release 🏆 was tested for a very long time to work well and have consistent performance and behavior characteristics.

## Supporting Tools
+ A utility main for parsing the hands pipeline of mediapipe into its nodes is [available for reuse](mediapipe_analysis/analysis/pipeline_parser.py) (applying the more advanced one from the liberation branch can be more helpful).
+ The input to the pipeline's head is just an ndarray image, [see here for discussion](https://github.com/nui-ai/mediapipe/blob/52e984567f30d8ffe79289e3852f8e9af2a6f69a/mediapipe/python/solution_base.py#L355-L372).
+ [a main running the same pipeline which the python api runs](hand_tracking_pipeline_run.py) while writing its output per frame as protobuf, since our liberation branch has mains which run a comparison of different pipeline runs which uses that format; 
  + its output data file can be used to compare results with liberated versions of the pipeline. this is just a repeatable way to get the original pipeline's results over a given input video file. this is necessary because the original example mains only ran the pipeline which the python api uses *wrapped* by other pipelines which made that pipeline's outputs inaccessible from the outside, so we couldn't just use them for that, so we added this main which mirrors the original python example but uses this pipeline directly (while writing out its output as protobuf).
  + this was instrumental in showing that the first step of liberation changes still yields identical results to the original pipeline down to the byte for each frame of a good not-short and diverse input video file!
  + to run it, enter the venv in which you built and issue e.g.: `python -P hand_tracking_pipeline_run.py sample-video-two-hands.avi`

## ReBuilding MediaPipe v0.10.13 Hand Tracking
[See the detailed build guide](BUILDING.md)

