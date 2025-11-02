//
// Created by matan on 9/30/25.
//

#ifndef MEDIAPIPE_DETECTION_MODEL_PROVIDER_H
#define MEDIAPIPE_DETECTION_MODEL_PROVIDER_H

#include <string>

#include "absl/status/status.h"
#include "mediapipe/framework/api2/node.h"
#include "mediapipe/framework/api2/port.h"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/resources.h"
namespace mediapipe_v01013_based::api2 {

    class DetectionModelProvider : public mediapipe_v01013_based::api2::Node {
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

}  // namespace mediapipe_v01013_based::api2

#endif //MEDIAPIPE_DETECTION_MODEL_PROVIDER_H