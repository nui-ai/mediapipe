#ifndef MEDIAPIPE_NUI_DESKTOP_CALCULATORS_RESOURCE_PROVIDER_CALCULATOR_NEW_H_
#define MEDIAPIPE_NUI_DESKTOP_CALCULATORS_RESOURCE_PROVIDER_CALCULATOR_NEW_H_

#include <string>

#include "absl/status/status.h"
#include "mediapipe/framework/api2/node.h"
#include "mediapipe/framework/api2/port.h"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/resources.h"
namespace mediapipe::api2 {

class DetectionModelProvider : public mediapipe::api2::Node {
 public:
  static constexpr api2::SideInput<std::string>::Multiple kIds{"RESOURCE_ID"};
  static constexpr api2::SideOutput<Resource>::Multiple kResources{"RESOURCE"};

  MEDIAPIPE_NODE_INTERFACE(DetectionModelProvider, kIds, kResources);

  static absl::Status UpdateContract(CalculatorContext* cc);

  absl::Status Open(CalculatorContext* cc) override;

  absl::Status Process(CalculatorContext* cc) override {
    return absl::OkStatus();
  }
};

}  // namespace mediapipe::api2

#endif  // MEDIAPIPE_NUI_DESKTOP_CALCULATORS_RESOURCE_PROVIDER_CALCULATOR_NEW_H_
