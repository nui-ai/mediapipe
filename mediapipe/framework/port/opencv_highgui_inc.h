// Copyright 2023 The MediaPipe Authors.
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

#ifndef MEDIAPIPE_FRAMEWORK_PORT_OPENCV_HIGHGUI_INC_H_
#define MEDIAPIPE_FRAMEWORK_PORT_OPENCV_HIGHGUI_INC_H_

#if __has_include(<opencv2/core/version.hpp>)
#include <opencv2/core/version.hpp>
#elif __has_include(<opencv4/opencv2/core/version.hpp>)
#include <opencv4/opencv2/core/version.hpp>
#else
#error "Cannot find OpenCV version.hpp header!"
#endif

#include "mediapipe/framework/port/opencv_core_inc.h"

#ifdef CV_VERSION_EPOCH  // for OpenCV 2.x
#if __has_include(<opencv2/highgui/highgui.hpp>)
#include <opencv2/highgui/highgui.hpp>
#elif __has_include(<opencv4/opencv2/highgui/highgui.hpp>)
#include <opencv4/opencv2/highgui/highgui.hpp>
#else
#error "Cannot find OpenCV highgui/highgui.hpp header!"
#endif
#else
#if __has_include(<opencv2/highgui.hpp>)
#include <opencv2/highgui.hpp>
#elif __has_include(<opencv4/opencv2/highgui.hpp>)
#include <opencv4/opencv2/highgui.hpp>
#else
#error "Cannot find OpenCV highgui.hpp header!"
#endif
#endif

#endif  // MEDIAPIPE_FRAMEWORK_PORT_OPENCV_HIGHGUI_INC_H_
