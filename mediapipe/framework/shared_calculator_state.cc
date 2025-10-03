#include "mediapipe/framework/shared_calculator_state.h"
#include <utility>

namespace mediapipe {

int SharedCalculatorState::counter_ = 0;
std::map<std::string, std::string> SharedCalculatorState::config_;
std::mutex SharedCalculatorState::mutex_;
std::vector<Anchor> SharedCalculatorState::anchors_;
std::shared_ptr<tflite::OpResolver> SharedCalculatorState::op_resolver_;

int SharedCalculatorState::GetCounter() {
    std::lock_guard<std::mutex> lock(mutex_);
    return counter_;
}
void SharedCalculatorState::IncrementCounter() {
    std::lock_guard<std::mutex> lock(mutex_);
    ++counter_;
}
void SharedCalculatorState::ResetCounter() {
    std::lock_guard<std::mutex> lock(mutex_);
    counter_ = 0;
}
const std::map<std::string, std::string>& SharedCalculatorState::GetConfig() {
    std::lock_guard<std::mutex> lock(mutex_);
    return config_;
}
void SharedCalculatorState::SetConfig(const std::map<std::string, std::string>& config) {
    std::lock_guard<std::mutex> lock(mutex_);
    config_ = config;
}
void SharedCalculatorState::SetAnchors(const std::vector<Anchor>& anchors) {
    std::lock_guard<std::mutex> lock(mutex_);
    anchors_ = anchors;
}
const std::vector<Anchor>& SharedCalculatorState::GetAnchors() {
    std::lock_guard<std::mutex> lock(mutex_);
    return anchors_;
}
void SharedCalculatorState::SetOpResolver(std::shared_ptr<tflite::OpResolver> op_resolver) {
    std::lock_guard<std::mutex> lock(mutex_);
    op_resolver_ = std::move(op_resolver);
}
std::shared_ptr<tflite::OpResolver> SharedCalculatorState::GetOpResolver() {
    std::lock_guard<std::mutex> lock(mutex_);
    return op_resolver_;
}

} // namespace mediapipe

