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

        static const std::map<std::string, std::string>& GetConfig();
        static void SetConfig(const std::map<std::string, std::string>& config);

    private:
        static int counter_;
        static std::map<std::string, std::string> config_;
        static std::mutex mutex_;
    };

}  // namespace mediapipe

#endif  // MEDIAPIPE_NUI_DESKTOP_SHARED_CALCULATOR_STATE_H_