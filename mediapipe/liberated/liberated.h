//
// Created by matan on 10/27/25.
//

#ifndef MEDIAPIPE_LIBERATED_H
#define MEDIAPIPE_LIBERATED_H

#include <memory>

#include "mediapipe/framework/memory_manager.h"
#include "mediapipe/framework/api2/port.h"
#include "mediapipe/calculators/tensor/image_to_tensor_utils.h"
#include "mediapipe/calculators/tensor/image_to_tensor_calculator_core.h"

namespace mediapipe {

// A simple helper that owns an ImageToTensorCalculatorCore instance.
// It constructs the core with the same options used by HeadCalculator::Open().
class Liberated {
 public:
  explicit Liberated(MemoryManager* memory_manager);
  ~Liberated() = default;

  // Non-copyable, movable.
  Liberated(const Liberated&) = delete;
  Liberated& operator=(const Liberated&) = delete;
  Liberated(Liberated&&) = default;
  Liberated& operator=(Liberated&&) = default;

 private:
  std::unique_ptr<api2::ImageToTensorCalculatorCore> image_to_tensor_core_;
  std::unique_ptr<ImageToTensorConverter> gpu_converter_;
  std::unique_ptr<ImageToTensorConverter> cpu_converter_;
};

}  // namespace mediapipe

#endif  // MEDIAPIPE_LIBERATED_H