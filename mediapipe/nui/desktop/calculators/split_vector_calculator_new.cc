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

#include "mediapipe/nui/desktop/calculators/split_vector_calculator_new.h"

#include <array>
#include <cstdint>

#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/formats/classification.pb.h"
#include "mediapipe/framework/formats/detection.pb.h"
#include "mediapipe/framework/formats/image.h"
#include "mediapipe/framework/formats/image_frame.h"
#include "mediapipe/framework/formats/landmark.pb.h"
#include "mediapipe/framework/formats/matrix.h"
#include "mediapipe/framework/formats/rect.pb.h"
#include "mediapipe/framework/formats/tensor.h"
#include "tensorflow/lite/interpreter.h"

#if !defined(MEDIAPIPE_DISABLE_GL_COMPUTE)
#include "tensorflow/lite/delegates/gpu/gl/gl_buffer.h"
#endif  // !defined(MEDIAPIPE_DISABLE_GL_COMPUTE)

namespace mediapipe {

typedef SplitVectorCalculator<TfLiteTensor, false>
    SplitTfLiteTensorVectorCalculatorNew;
REGISTER_CALCULATOR(SplitTfLiteTensorVectorCalculatorNew);

typedef SplitVectorCalculator<Tensor, true> SplitTensorVectorCalculatorNew;
REGISTER_CALCULATOR(SplitTensorVectorCalculatorNew);

typedef SplitVectorCalculator<mediapipe::NormalizedLandmark, false>
    SplitLandmarkVectorCalculatorNew;
REGISTER_CALCULATOR(SplitLandmarkVectorCalculatorNew);

typedef SplitVectorCalculator<mediapipe::NormalizedLandmarkList, false>
    SplitNormalizedLandmarkListVectorCalculatorNew;
REGISTER_CALCULATOR(SplitNormalizedLandmarkListVectorCalculatorNew);

typedef SplitVectorCalculator<mediapipe::NormalizedRect, false>
    SplitNormalizedRectVectorCalculatorNew;
REGISTER_CALCULATOR(SplitNormalizedRectVectorCalculatorNew);

typedef SplitVectorCalculator<Matrix, false> SplitMatrixVectorCalculatorNew;
REGISTER_CALCULATOR(SplitMatrixVectorCalculatorNew);

#if !defined(MEDIAPIPE_DISABLE_GL_COMPUTE)
typedef SplitVectorCalculator<::tflite::gpu::gl::GlBuffer, true>
    MovableSplitGlBufferVectorCalculator;
REGISTER_CALCULATOR(MovableSplitGlBufferVectorCalculator);
#endif


}  // namespace mediapipe
