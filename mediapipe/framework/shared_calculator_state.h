//
// Created by matan on 9/30/25.
//

#ifndef MEDIAPIPE_NUI_DESKTOP_SHARED_CALCULATOR_STATE_H_
#define MEDIAPIPE_NUI_DESKTOP_SHARED_CALCULATOR_STATE_H_

#include <map>
#include <string>
#include <mutex>
#include "mediapipe/framework/formats/rect.pb.h"

namespace hand_tracking_mp_lean {
    // Forward declaration to break circular dependency
    class Image;
    class SharedCalculatorState {
    public:

        static int GetCounter();
        static void IncrementCounter();
        static void ResetCounter();

        const uint32_t NUM_HANDS = 3; // only used by a calculator we no longer actively trigger by now

        std::vector<::hand_tracking_mp_lean::NormalizedRect> prev_hand_rects_from_landmarks;

    private:
        static int counter_;
        static std::mutex mutex_;
    };


}  // namespace hand_tracking_mp_lean

#endif  // MEDIAPIPE_NUI_DESKTOP_SHARED_CALCULATOR_STATE_H_