//
// Created by matan on 9/30/25.
//

#ifndef MEDIAPIPE_NUI_DESKTOP_SHARED_CALCULATOR_STATE_H_
#define MEDIAPIPE_NUI_DESKTOP_SHARED_CALCULATOR_STATE_H_

#include <map>
#include <string>
#include <mutex>
#include <vector>
#include <memory>


#include "mediapipe/calculators/tflite/tflite_custom_op_resolver_calculator.pb.h"
#include "tensorflow/lite/core/api/op_resolver.h"
#include "mediapipe/framework/formats/object_detection/anchor.pb.h"
#include "tensorflow/lite/core/api/op_resolver.h"

namespace mediapipe {

    class SharedCalculatorState {
    public:
        static int GetCounter();
        static void IncrementCounter();
        static void ResetCounter();

        static const std::map<std::string, std::string>& GetConfig();
        static void SetConfig(const std::map<std::string, std::string>& config);

        // Complex side packet storage
        static void SetAnchors(const std::vector<mediapipe::Anchor>& anchors);
        static const std::vector<mediapipe::Anchor>& GetAnchors();
        static void SetOpResolver(std::shared_ptr<tflite::OpResolver> op_resolver);
        static std::shared_ptr<tflite::OpResolver> GetOpResolver();

    private:
        static int counter_;
        static std::map<std::string, std::string> config_;
        static std::mutex mutex_;

        static std::vector<mediapipe::Anchor> anchors_;
        static std::shared_ptr<tflite::OpResolver> op_resolver_;
    };

}  // namespace mediapipe

#endif  // MEDIAPIPE_NUI_DESKTOP_SHARED_CALCULATOR_STATE_H_