// Copyright 2019 The MediaPipe Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef MEDIAPIPE_FRAMEWORK_PORT_OPENCV_CALIB3D_INC_H_
#define MEDIAPIPE_FRAMEWORK_PORT_OPENCV_CALIB3D_INC_H_

#if __has_include(<opencv2/core/version.hpp>)
#include <opencv2/core/version.hpp>
#elif __has_include(<opencv4/opencv2/core/version.hpp>)
#include <opencv4/opencv2/core/version.hpp>
#else
#error "Cannot find OpenCV version.hpp header!"
#endif

#ifdef CV_VERSION_EPOCH  // for OpenCV 2.x
#if __has_include(<opencv2/calib3d/calib3d.hpp>)
#include <opencv2/calib3d/calib3d.hpp>
#elif __has_include(<opencv4/opencv2/calib3d/calib3d.hpp>)
#include <opencv4/opencv2/calib3d/calib3d.hpp>
#else
#error "Cannot find OpenCV calib3d/calib3d.hpp header!"
#endif
#else
#if __has_include(<opencv2/calib3d.hpp>)
#include <opencv2/calib3d.hpp>
#elif __has_include(<opencv4/opencv2/calib3d.hpp>)
#include <opencv4/opencv2/calib3d.hpp>
#else
#error "Cannot find OpenCV calib3d.hpp header!"
#endif
#endif

#endif  // MEDIAPIPE_FRAMEWORK_PORT_OPENCV_CALIB3D_INC_H_
