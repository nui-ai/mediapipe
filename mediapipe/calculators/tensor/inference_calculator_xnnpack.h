// Copyright 2022 The MediaPipe Authors.
#ifndef MEDIAPIPE_CALCULATORS_TENSOR_INFERENCE_CALCULATOR_XNNPACK_H_
#define MEDIAPIPE_CALCULATORS_TENSOR_INFERENCE_CALCULATOR_XNNPACK_H_

#include <memory>
#include <string>
#include <vector>
#include "mediapipe/calculators/tensor/inference_calculator.h"
#include "mediapipe/calculators/tensor/tensor_span.h"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/formats/tensor.h"

namespace mediapipe {
namespace api2 {

class InferenceCalculatorXnnpackImpl
    : public InferenceCalculatorNodeImpl<InferenceCalculatorXnnpack,
                                         InferenceCalculatorXnnpackImpl> {
 public:
  static absl::Status UpdateContract(CalculatorContract* cc);
  absl::Status Open(CalculatorContext* cc) override;
  absl::Status Close(CalculatorContext* cc) override;

 protected:
  // Subclasses must override this to provide the model path.
  [[nodiscard]] virtual std::string GetModelPath() const = 0;

 private:
  absl::StatusOr<std::vector<Tensor>> Process(
      CalculatorContext* cc, const TensorSpan& tensor_span) override;
  absl::StatusOr<std::unique_ptr<InferenceRunner>> CreateInferenceRunner(
      CalculatorContext* cc);
  absl::StatusOr<TfLiteDelegatePtr> CreateDelegate(CalculatorContext* cc);

  std::unique_ptr<InferenceRunner> inference_runner_;
};

}  // namespace api2
}  // namespace mediapipe

#endif  // MEDIAPIPE_CALCULATORS_TENSOR_INFERENCE_CALCULATOR_XNNPACK_H_

