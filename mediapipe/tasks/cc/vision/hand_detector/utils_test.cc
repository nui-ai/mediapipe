#include "mediapipe/tasks/cc/vision/hand_detector/utils.h"

#include "absl/status/statusor.h"
#include "mediapipe/calculators/tensor/tensors_to_detections_calculator.pb.h"
#include "mediapipe/calculators/tflite/ssd_anchors_calculator.pb.h"
#include "mediapipe/framework/deps/file_path.h"
#include "mediapipe/framework/port/file_helpers.h"
#include "mediapipe/framework/port/gmock.h"
#include "mediapipe/framework/port/gtest.h"
#include "mediapipe/tasks/cc/core/model_resources.h"
#include "mediapipe/tasks/cc/core/proto/external_file.pb.h"
#include "mediapipe/tasks/cc/vision/hand_detector/proto/hand_detector_graph_options.pb.h"
#include "mediapipe/tasks/cc/vision/utils/image_tensor_specs.h"
#include "research/aimatter/api/metadata_utils.h"

namespace hand_tracking_mp_lean::tasks::vision::hand_detector {
namespace {

namespace rapi = ::research::aimatter::api;

using ::hand_tracking_mp_lean::file::JoinPath;
using ::hand_tracking_mp_lean::tasks::core::ModelResources;
using ::hand_tracking_mp_lean::tasks::core::proto::ExternalFile;

constexpr char kTestDataDirectory[] = "/mediapipe/tasks/testdata/vision/";
constexpr char kTestModelResourcesTag[] = "test_model_resources";
constexpr char kModelWithMetadataName[] = "palm_detection_full.tflite";
constexpr float kEpsilon = 1e-6;

// Helper function to get ModelResources.
absl::StatusOr<std::unique_ptr<ModelResources>> CreateModelResourcesForModel(
    absl::string_view model_name) {
  auto external_file = std::make_unique<ExternalFile>();
  external_file->set_file_name(JoinPath("./", kTestDataDirectory, model_name));
  return ModelResources::Create(kTestModelResourcesTag,
                                std::move(external_file));
}

TEST(Utils, ConfigureSsdAnchorsCalculator) {
  MP_ASSERT_OK_AND_ASSIGN(auto model_resources,
                          CreateModelResourcesForModel(kModelWithMetadataName));
  const tflite::Model& model = *(model_resources->GetTfLiteModel());
  MP_ASSERT_OK_AND_ASSIGN(
      const auto metadata_fb,
      rapi::VerifyAndLoadMetadata<rapi::fb::FaceDetectorMetadata>(
          model, rapi::fb::FaceDetectorMetadataIdentifier()));
  hand_tracking_mp_lean::SsdAnchorsCalculatorOptions ssd_anchors_options;
  MP_ASSERT_OK(ConfigureSsdAnchorsCalculator(
      *BuildInputImageTensorSpecs(*model_resources), *metadata_fb,
      ssd_anchors_options));
  EXPECT_EQ(ssd_anchors_options.fixed_anchors().size(), 2016);
}

TEST(Utils, ConfigureTensorsToDetectionCalculator) {
  MP_ASSERT_OK_AND_ASSIGN(auto model_resources,
                          CreateModelResourcesForModel(kModelWithMetadataName));
  hand_tracking_mp_lean::TensorsToDetectionsCalculatorOptions tensors_to_detections_options;
  MP_ASSERT_OK(ConfigureTensorsToDetectionsCalculator(
      *BuildInputImageTensorSpecs(*model_resources), 2016, 0.1,
      tensors_to_detections_options));
  EXPECT_NEAR(tensors_to_detections_options.x_scale(), 192, kEpsilon);
  EXPECT_NEAR(tensors_to_detections_options.y_scale(), 192, kEpsilon);
  EXPECT_NEAR(tensors_to_detections_options.w_scale(), 192, kEpsilon);
  EXPECT_NEAR(tensors_to_detections_options.h_scale(), 192, kEpsilon);
  EXPECT_NEAR(tensors_to_detections_options.min_score_thresh(), 0.1, kEpsilon);
}

}  // namespace
}  // namespace hand_tracking_mp_lean::tasks::vision::hand_detector
