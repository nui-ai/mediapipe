//
// Created by matan on 9/30/25.
//

#ifndef MEDIAPIPE_NUI_DESKTOP_SHARED_CALCULATOR_STATE_H_
#define MEDIAPIPE_NUI_DESKTOP_SHARED_CALCULATOR_STATE_H_

#include <map>
#include <string>
#include <mutex>
#include "mediapipe/framework/formats/rect.pb.h"
#include "mediapipe/framework/formats/image.h"

namespace mediapipe {
    class SharedCalculatorState {
    public:

        static int GetCounter();
        static void IncrementCounter();
        static void ResetCounter();

        const uint32_t NUM_HANDS = 2;
        const uint32_t model_complexity = 1;
        const bool USE_PREV_LANDMARKS = true;

        // image being input to the pipeline
        std::shared_ptr<Image> image;
        // image to apply palm detection to, or null pointer when palm detection should be skipped
        std::shared_ptr<Image> palm_detection_image;

        // rects from landmarks tracking in the previous frame
        std::vector<::mediapipe::NormalizedRect> prev_hand_rects_from_landmarks;
    private:
        static int counter_;
        static std::mutex mutex_;
    };


}  // namespace mediapipe

#endif  // MEDIAPIPE_NUI_DESKTOP_SHARED_CALCULATOR_STATE_H_