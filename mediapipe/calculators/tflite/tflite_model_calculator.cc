// Copyright 2020 The MediaPipe Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <tuple>
#include <utility>

#include "absl/status/status.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "mediapipe/calculators/util/resource_provider_calculator.h"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/packet.h"
#include "mediapipe/framework/port/ret_check.h"
#include "mediapipe/framework/port/status_macros.h"
#include "mediapipe/framework/resources.h"
#include "tensorflow/lite/allocation.h"
#include "tensorflow/lite/model_builder.h"

namespace hand_tracking_mp_lean {

// Struct to hold onto resources so that they can be properly managed
// in the deleter function for TfLiteModel
struct ModelResources {
  ModelResources(Packet model_packet, std::unique_ptr<Resource> res = nullptr)
      : packet(std::move(model_packet)), resource(std::move(res)) {}

  // These fields need to stay alive as long as the TFLite model uses them
  Packet packet;
  std::unique_ptr<Resource> resource;
};

// Loads as a TfLite model the model given as input side packet and outputs the loaded model.
//
// Input side packets:
//   MODEL_PATH - TfLite model file path as std::string. The model will be loaded
//                directly from the specified path.
//   MODEL_RESOURCE - TfLite model file as hand_tracking_mp_lean::Resource - enables
//                    managed, unmanaged, in-memory, mmaped resources.
//   MODEL_BLOB - TfLite model blob/file-contents (std::string). You can read
//                model blob from file (using whatever APIs you have) and pass
//                it to the graph as input side packet or you can use some of
//                calculators like LocalFileContentsCalculator to get model
//                blob and use it as input here.
//   MODEL_FD   - Tflite model file descriptor std::tuple<int, size_t, size_t>
//                containing (fd, offset, size).
//   MODEL_SPAN - TfLite model file contents in absl::Span<const uint8_t>, whose
//                underline buffer is owned outside of this calculator. User can
//                get the model span from a managed environment and pass it to
//                the graph as input side packet.
//
// Output side packets:
//   MODEL - TfLite model. (std::unique_ptr<tflite::FlatBufferModel,
//           std::function<void(tflite::FlatBufferModel*)>>)
//   SHARED_MODEL - TfLite model (std::shared_ptr<tflite::FlatBufferModel>) to
//           be shared by multiple downstream calculators.
//
// Example use:
//
// node {
//   calculator: "TfLiteModelCalculator"
//   input_side_packet: "MODEL_PATH:model_path"
//   output_side_packet: "MODEL:model"
// }
//
// node {
//   calculator: "TfLiteModelCalculator"
//   input_side_packet: "MODEL_RESOURCE:model_resource"
//   output_side_packet: "MODEL:model"
// }
//
class TfLiteModelCalculator : public CalculatorBase {
 public:
  using TfLiteModelPtr =
      std::unique_ptr<tflite::FlatBufferModel,
                      std::function<void(tflite::FlatBufferModel*)>>;
  using SharedTfLiteModelPtr = std::shared_ptr<tflite::FlatBufferModel>;

  static constexpr absl::string_view kModelPathTag = "MODEL_PATH";
  static constexpr absl::string_view kModelResourceTag = "MODEL_RESOURCE";
  static constexpr absl::string_view kModelSpanTag = "MODEL_SPAN";
  static constexpr absl::string_view kModelBlobTag = "MODEL_BLOB";
  static constexpr absl::string_view kModelFDTag = "MODEL_FD";
  static constexpr absl::string_view kModelTag = "MODEL";
  static constexpr absl::string_view kSharedModelTag = "SHARED_MODEL";

  static absl::Status GetContract(CalculatorContract* cc) {
    if (cc->InputSidePackets().HasTag(kModelPathTag)) {
      cc->InputSidePackets().Tag(kModelPathTag).Set<std::string>();
    }

    if (cc->InputSidePackets().HasTag(kModelBlobTag)) {
      cc->InputSidePackets().Tag(kModelBlobTag).Set<std::string>();
    }

    if (cc->InputSidePackets().HasTag(kModelFDTag)) {
      cc->InputSidePackets()
          .Tag(kModelFDTag)
          .Set<std::tuple<int, size_t, size_t>>();
    }

    if (cc->InputSidePackets().HasTag(kModelSpanTag)) {
      cc->InputSidePackets()
          .Tag(kModelSpanTag)
          .Set<absl::Span<const uint8_t>>();
    }

    if (cc->InputSidePackets().HasTag(kModelResourceTag)) {
      cc->InputSidePackets().Tag(kModelResourceTag).Set<Resource>();
    }

    RET_CHECK(cc->OutputSidePackets().HasTag(kModelTag) ^
              cc->OutputSidePackets().HasTag(kSharedModelTag));

    if (cc->OutputSidePackets().HasTag(kModelTag)) {
      cc->OutputSidePackets().Tag(kModelTag).Set<TfLiteModelPtr>();
    } else if (cc->OutputSidePackets().HasTag(kSharedModelTag)) {
      cc->OutputSidePackets().Tag(kSharedModelTag).Set<SharedTfLiteModelPtr>();
    }

    return absl::OkStatus();
  }

  absl::Status Open(CalculatorContext* cc) override {
    Packet model_packet;
    std::unique_ptr<tflite::FlatBufferModel> model;
    std::unique_ptr<Resource> resource;

    // Load model from file path using ResourceProviderCalculator::LoadModelFromPath
    if (cc->InputSidePackets().HasTag(kModelPathTag)) {
      model_packet = cc->InputSidePackets().Tag(kModelPathTag);
      RET_CHECK(!model_packet.IsEmpty());
      const std::string& model_path = model_packet.Get<std::string>();

      // Use ResourceProviderCalculator to load the model from path
      auto resource_or = api2::ResourceProviderCalculator::LoadModelFromPath(model_path);
      RET_CHECK(resource_or.ok()) << "Failed to load model from path: " << model_path;

      resource = std::move(resource_or).value();
      absl::string_view model_view = resource->ToStringView();
      model = tflite::FlatBufferModel::BuildFromBuffer(model_view.data(),
                                                     model_view.size());
      ABSL_LOG(INFO) << "tflite model loaded from file path";
    }

    if (cc->InputSidePackets().HasTag(kModelBlobTag)) {
      model_packet = cc->InputSidePackets().Tag(kModelBlobTag);
      RET_CHECK(!model_packet.IsEmpty());
      const std::string& model_blob = model_packet.Get<std::string>();
      model = tflite::FlatBufferModel::BuildFromBuffer(model_blob.data(),
                                                       model_blob.size());
    }

    if (cc->InputSidePackets().HasTag(kModelSpanTag)) {
      model_packet = cc->InputSidePackets().Tag(kModelSpanTag);
      RET_CHECK(!model_packet.IsEmpty());
      const absl::Span<const uint8_t>& model_view =
          model_packet.Get<absl::Span<const uint8_t>>();
      model = tflite::FlatBufferModel::BuildFromBuffer(
          reinterpret_cast<const char*>(model_view.data()), model_view.size());
    }

    if (cc->InputSidePackets().HasTag(kModelResourceTag)) {
      model_packet = cc->InputSidePackets().Tag(kModelResourceTag);
      RET_CHECK(!model_packet.IsEmpty());
      absl::string_view model_view =
          model_packet.Get<Resource>().ToStringView();
      model = tflite::FlatBufferModel::BuildFromBuffer(model_view.data(),
                                                       model_view.size());
      ABSL_LOG(INFO) << "tflite model loaded from resource object";
    }

    if (cc->InputSidePackets().HasTag(kModelFDTag)) {
#if defined(ABSL_HAVE_MMAP) && !TFLITE_WITH_STABLE_ABI
      model_packet = cc->InputSidePackets().Tag(kModelFDTag);
      const auto& model_fd =
          model_packet.Get<std::tuple<int, size_t, size_t>>();
      auto model_allocation = std::make_unique<tflite::MMAPAllocation>(
          std::get<0>(model_fd), std::get<1>(model_fd), std::get<2>(model_fd),
          tflite::DefaultErrorReporter());
      model = tflite::FlatBufferModel::BuildFromAllocation(
          std::move(model_allocation), tflite::DefaultErrorReporter());
#else
      return absl::FailedPreconditionError(
          "Loading by file descriptor is not supported on this platform.");
#endif
    }

    RET_CHECK(model) << "Failed to load TfLite model.";

    // Create a shared_ptr to manage the resources properly
    auto resources = std::make_shared<ModelResources>(model_packet, std::move(resource));

    // Set up a deleter that will keep the resources alive along with the model
    auto deleter = [resources](tflite::FlatBufferModel* model) {
      delete model;  // Delete the model first
      // The resources shared_ptr will be deleted when it goes out of scope
    };

    TfLiteModelPtr output_model(model.release(), deleter);

    if (cc->OutputSidePackets().HasTag(kModelTag)) {
      cc->OutputSidePackets().Tag(kModelTag).Set(
          MakePacket<TfLiteModelPtr>(std::move(output_model)));
    } else {
      cc->OutputSidePackets()
          .Tag(kSharedModelTag)
          .Set(MakePacket<SharedTfLiteModelPtr>(std::move(output_model)));
    }

    return absl::OkStatus();
  }

  absl::Status Process(CalculatorContext* cc) override {
    return absl::OkStatus();
  }
};
REGISTER_CALCULATOR(TfLiteModelCalculator);

}  // namespace hand_tracking_mp_lean
