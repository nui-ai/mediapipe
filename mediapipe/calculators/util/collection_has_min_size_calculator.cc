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

#include <vector>

#include "mediapipe/calculators/util/collection_has_min_size_calculator.h"
#include "mediapipe/calculators/tensor/image_to_tensor_utils.h"
#include "mediapipe/framework/api2/port.h"
#include "mediapipe/framework/port/status.h"

#include "mediapipe/framework/formats/classification.pb.h"
#include "mediapipe/framework/formats/landmark.pb.h"
#include "mediapipe/framework/formats/rect.pb.h"

namespace mediapipe_v01013_based {

typedef CollectionHasMinSizeCalculator<
    std::vector<mediapipe_v01013_based::NormalizedLandmarkList>>
    NormalizedLandmarkListVectorHasMinSizeCalculator;
REGISTER_CALCULATOR(NormalizedLandmarkListVectorHasMinSizeCalculator);

typedef CollectionHasMinSizeCalculator<
    std::vector<mediapipe_v01013_based::ClassificationList>>
    ClassificationListVectorHasMinSizeCalculator;
REGISTER_CALCULATOR(ClassificationListVectorHasMinSizeCalculator);

}  // namespace mediapipe_v01013_based
