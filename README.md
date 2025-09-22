# Mediapipe Hand Tracking Radical Improvement Track

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
+ A utility main for parsing the hands pipeline of mediapipe into its nodes is [available for reuse](mediapipe_analysis/analysis/pipeline_parser.py).
+ The input to the pipeline's head is just an ndarray image, [see here for discussion](https://github.com/nui-ai/mediapipe/blob/52e984567f30d8ffe79289e3852f8e9af2a6f69a/mediapipe/python/solution_base.py#L355-L372).

## ReBuilding MediaPipe v0.10.13 Hand Tracking
[See the detailed build guide](BUILDING.md)

