//
// Created by matan on 9/30/25.
//

#ifndef MEDIAPIPE_NUI_DESKTOP_SHARED_CALCULATOR_STATE_H_
#define MEDIAPIPE_NUI_DESKTOP_SHARED_CALCULATOR_STATE_H_

#include <map>
#include <string>
#include <mutex>

namespace mediapipe {

    class SharedCalculatorState {
    public:

        static int GetCounter();
        static void IncrementCounter();
        static void ResetCounter();

        uint32_t num_hands = 2;
        uint32_t model_complexity = 1;
        bool USE_PREV_LANDMARKS = true;

        bool prev_has_enough_hands;

    private:
        static int counter_;
        static std::mutex mutex_;
    };

}  // namespace mediapipe

#endif  // MEDIAPIPE_NUI_DESKTOP_SHARED_CALCULATOR_STATE_H_