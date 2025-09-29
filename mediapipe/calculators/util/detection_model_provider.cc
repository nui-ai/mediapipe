//
// Created by matan on 9/30/25.
//

#include "detection_model_provider.h"

#include <memory>
#include <utility>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "mediapipe/calculators/util/resource_provider_calculator.pb.h"
#include "mediapipe/framework/api2/node.h"
#include "mediapipe/framework/api2/packet.h"
#include "mediapipe/framework/api2/port.h"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/port/ret_check.h"
#include "mediapipe/framework/port/status_macros.h"
#include "mediapipe/framework/resources.h"

namespace mediapipe::api2 {

    absl::Status DetectionModelProvider::Open(CalculatorContext* cc) {
        // Always load the fixed model file, ignore side packets and options.
        constexpr absl::string_view kModelPath = "mediapipe/modules/palm_detection/palm_detection_full.tflite";
        Resources::Options res_opts = {};
        // Default to binary mode for tflite.
        res_opts.read_as_binary = true;

        MP_ASSIGN_OR_RETURN(std::unique_ptr<Resource> res,
                            cc->GetResources().Get(kModelPath, res_opts));
        Packet<Resource> res_packet = api2::PacketAdopting(std::move(res));
        kResources(cc)[0].Set(std::move(res_packet));
        return absl::OkStatus();
    }

    MEDIAPIPE_REGISTER_NODE(DetectionModelProvider)

    }