//
// Created by matan on 9/30/25.
//

#include "shared_calculator_state.h"

#include <map>
#include <string>
#include <mutex>


namespace hand_tracking_mp_lean {

    // global value definitions which previously came as input to the entire invocation of the root graph
    int SharedCalculatorState::counter_ = 0;
    std::mutex SharedCalculatorState::mutex_;

    // a frame number counter
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

    // void SharedCalculatorState::SetConfig(const std::map<std::string, std::string>& config) {
    //     std::lock_guard<std::mutex> lock(mutex_);
    //     config_ = config;
    // }

    // each mediapipe stream definition pattern (TAG:name, name, CLONE:n:name etc.) and the framework's wiring
    // of input streams to output streams from other nodes should be made a simple typed definition of the "packet"
    // type of the stream, which both producer and consumer should use.
    // this avoids the crufty calculator-opinionated way of consuming the input streams
    // into a simple access of the replacement member here ― which is all we need ― consumer reads it,
    // producer writes/updates it. no concurrency issues even under framework begin-end node looping.
    // why bother? (a) this untangles from the framework so when things are no longer calculators,
    // they will just keep working, which is major. (b) all messages passed in one place mildly helps
    // redistributing plain function flow, it abstracts over wiring argument passing which mildly
    // smooths out merging and refactoring.
    std::pair<int, int> image_size_;

}  // namespace hand_tracking_mp_lean