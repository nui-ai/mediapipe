# Mediapipe Hand Tracking Radical Improvement Track

This branch represents the attempt of building a separate hands pipeline cloned (and deep-renamed to avoid conflicts) such that we can run both the original v0.10.13 code level with the original pipeline, and the new pipeline clone side-by-side while comparing them by graph nodes which would use google's absl differencer. This was not a vialbe development path for liberating the pipeline because:

+ For working (modifying code to our purpose) deep cloning at many layers of mediapipe code emerged as necessary.  
+ The level of build friction in making such deep going clones buildable is way high, making any development iteration almost impossible: every new source and sub-graph added need to be maticulously punched into the build by hand in observerence of the source paths and hierarchy, only being able to drive this stitching by repeatedly running the build and interpreting its long often obtuse error messages. the paths are long-nested and source files for working on a feature are interspersed in many places, enough to make this almost unhuman to work in. 
+ Cloning source files at multiple levels of the mediapipe source tree is utterly confusing and error-prone, even without the build friction.

So that strategy could have only worked for a small project or one which does not rely on manual build adaptations for every change such as with the C++ project we have here.<br> 

This is a pity because it means we can't co-exist with the original pipeline in the same build, and hence not in the same binary and running process, for smooth parity testing with such simple comparator graph nodes as planned.