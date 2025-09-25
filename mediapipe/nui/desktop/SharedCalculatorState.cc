#include <map>
#include <string>
#include <mutex>
#include "mediapipe/nui/desktop/SharedCalculatorState.h"

namespace mediapipe {

// Static member definitions
int SharedCalculatorState::counter_ = 0;
std::map<std::string, std::string> SharedCalculatorState::config_;
std::mutex SharedCalculatorState::mutex_;

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
  return config_;
}

void SharedCalculatorState::SetConfig(const std::map<std::string, std::string>& config) {
  std::lock_guard<std::mutex> lock(mutex_);
  config_ = config;
}

}  // namespace mediapipe
