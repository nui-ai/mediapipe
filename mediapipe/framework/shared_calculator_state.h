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

        const uint32_t NUM_HANDS = 3;

        std::vector<::mediapipe_v01013_based::NormalizedRect> prev_hand_rects_from_landmarks;

    private:
        static int counter_;
        static std::mutex mutex_;
    };


}  // namespace mediapipe_v01013_based

#endif  // MEDIAPIPE_NUI_DESKTOP_SHARED_CALCULATOR_STATE_H_