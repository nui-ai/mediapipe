//
// Created by matan on 9/30/25.
//

#ifndef MEDIAPIPE_NUI_DESKTOP_SHARED_CALCULATOR_STATE_H_
#define MEDIAPIPE_NUI_DESKTOP_SHARED_CALCULATOR_STATE_H_

#include <map>
#include <string>
#include <mutex>
#include "mediapipe/framework/formats/rect.pb.h"

namespace mediapipe_v01013_based {
    // Forward declaration to break circular dependency
    class Image;
    class SharedCalculatorState {
    public:

        static int GetCounter();
        static void IncrementCounter();
        static void ResetCounter();

        const uint32_t NUM_HANDS = 2;
        const bool USE_PREV_LANDMARKS = true;

        // image being input to the pipeline
        std::shared_ptr<const mediapipe_v01013_based::Image> image;
        // image to apply palm detection to, or null pointer when palm detection should be skipped
        std::shared_ptr<const mediapipe_v01013_based::Image> palm_detection_image;

        // rects from landmarks tracking in the previous frame
        std::vector<::mediapipe_v01013_based::NormalizedRect> prev_hand_rects_from_landmarks;

        uint32_t ImageToTensorCalculatorOptions_input_selection_field;  //  0 => use palm_detection_image as input;
        int32_t ImageToTensorCalculatorOptions_num_landmakrs;
        int32_t ImageToTensorCalculatorOptions_output_tensor_width ;
        int32_t ImageToTensorCalculatorOptions_output_tensor_height;
        bool ImageToTensorCalculatorOptions_keep_aspect_ratio;
        float ImageToTensorCalculatorOptions_float_range_min;
        float ImageToTensorCalculatorOptions_float_range_max;
        int32_t ImageToTensorCalculatorOptions_border_mode;
        float ImageToTensorCalculatorOptions_normalize_z;

    private:
        static int counter_;
        static std::mutex mutex_;
    };


}  // namespace mediapipe_v01013_based

#endif  // MEDIAPIPE_NUI_DESKTOP_SHARED_CALCULATOR_STATE_H_