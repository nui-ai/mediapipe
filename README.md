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
+ A utility main for parsing the hands pipeline of mediapipe into its nodes is [available for reuse](mediapipe_analysis/analysis/pipeline_parser.py).
+ The input to the pipeline's head is just an ndarray image, [see here for discussion](https://github.com/nui-ai/mediapipe/blob/52e984567f30d8ffe79289e3852f8e9af2a6f69a/mediapipe/python/solution_base.py#L355-L372).

## ReBuilding MediaPipe v0.10.13 Hand Tracking
[See the detailed build guide](BUILDING.md)

## About Mediapipe Framework Code Patterns

Mediapipe framework code (and its use in specific calculators, graphs, and solutions) often involves multiple layers of abstraction and "fluff" that can obscure the core logic of what is being implemented. This is especially true for calculators that perform the bottom-line operations happening on a each input tick fed to their input streams. so for example the mere feat of splitting a vector into multiple parts will constitute more invoking of layers of abstraction than the actual final operatinon being performed:


1. Calculator Framework Layer:<BR>
e.g. MediaPipe's CalculatorBase and contract system, requiring input/output stream setup and option parsing.

2. Options/Configuration Layer:<BR>
Relies on SplitVectorCalculatorOptions protobuf for specifying split ranges and behaviors (element_only, combine_outputs).

3. Type Traits Layer:<BR>
Uses C++ type traits (is_copy_constructible, is_move_constructible) to select copy/move logic and enforce constraints.

4. Error Checking Layer:<BR>
Extensive use of RET_CHECK, RET_CHECK_OK, and error returns to validate configuration and input correctness.

5. Output Type Selection Layer:<BR>
Dynamically sets output types based on options, supporting both element and vector outputs.

6. Template Specialization Layer<BR>
ses template specialization to handle copyable vs. movable types, with separate code paths and error handling.

7. Packet/Stream Abstraction Layer<BR>
Wraps outputs in MediaPipe packets and streams, rather than returning raw vectors.

8. Hyper-modularity
In addition, the grain-of-sand level approach of modularity entails layers abstracting over different memory management strategies (copy vs. move semantics).

9. Status/Result Layer<BR>
Returns absl::Status for all operations, propagating errors through the framework. 

All these layers make mediapipe code modular down to grain of sand, and provide safety, but also make it significantly complex for a developer to review for things as small as a vector splitting function, until you get familiar with all those layers to be able to skip them by glossing when you're not interested, because:
+ not all of these layers are abstractions but rather they explicitly get invoked in leafy layers of the C++ code.  
+ a large percent of the code is safety asserting (using this macro or another) things that the layers of abstraction (especially in a typed language) should already guarantee, providing no value but obfuscate the flow of code more. 

The same goes for why and how Bazel is used for building classes from protobuf definitions as a foundation for much of how the code is organized for multi-lingual use.

Explaining the concepts of synchronized processing in stream-processing, which the mediapipe framework forces calculators code into, is way beyond the scope of a readme, but you can best learn them by informed ChatGPT conversations, as the official documentation stops really only half-way.



