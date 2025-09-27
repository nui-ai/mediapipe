# Mediapipe Hand Tracking Radical Improvement Track

## Current Branch Objective
+ This branch bases off branch "buildable-reference" and such starts as a buildable mediapipe v0.10.13 code level. 
+ It is used for gradually "liberating" the pipeline into a form where the per-image results are obtained from code that runs in a plain C++ execution architecture (not a mediapipe graph) are the same as those obtained from the original v0.10.13 mediapipe graph which can be run by the "buildable-reference" branch. 
+ A fine distinction: the "liberated" outcome may still use some mediapipe api, but should not run as a _mediapipe pipeline_.
+ This branch is meant for active development! 
  + This branch is meant for active development where you freely modify the mediapipe codebase as needed on your way to a liberated pipeline, which you will likely do since many simple changes to the computation nodes go deep into mediapipe framework code ― which you can freely modify in this branch ― rather than deep cloning classes you'd like to change which will get you nowhere. That's because every new source file needs careful bazel build integration which is very tedious and often doesn't converge with sanity, so modifying mediapipe code directly is the only practical way to go. 
  + Rationale: there are no sane ways to have two versions of the same codebase in the same running process, esp. not with a bazel-based elaborate build like mediapipe; you cannot somehow expect that to happen unless you have tinkered object files for 20 years and even then ... don't try that. so don't worry about twisting the source for the purpose, the source code of this branch is not expected to also somehow serve the running of the original mediapipe graph.
  + If you are doubtful about the last comment, consider that it's a two day journey to learn from experience why you cannot expect to develop the liberation objective while preserving the existing pipeline, that kind of journey will hit into the great wall of deep cloning and build-integrating by hand for every change ― which might converge if you worked with Bazel C++ Protobuf based projects for 20 years, but otherwise is a dead end due to the lack of tooling to facilitate smooth paths at that.
  + If you wish to have branced-and-merge workflow, just branch off this branch while maintaining the "liberation" prefix in your work branches, treating this branch as the main of this work. 

So the development strategy here should be ― twist the entire mediapipe codebase to gradually un-pipeline the hand tracking pipeline, until finally you are left with code that accomplishes the same as the original pipeline, but without being a mediapipe pipeline! Eventually more work can remove any use of any mediapipe class and so on, but that's secondary to just running with complete parity without being a pipeline. 

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



