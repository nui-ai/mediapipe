#include <cstdint>
#include <utility>

#include "absl/status/status.h"
#include "mediapipe/framework/calculator_runner.h"
#include "mediapipe/framework/formats/image_frame.h"
#include "mediapipe/framework/packet.h"
#include "mediapipe/framework/port/gtest.h"
#include "mediapipe/framework/port/parse_text_proto.h"
#include "mediapipe/framework/port/status_matchers.h"
#include "mediapipe/framework/timestamp.h"

namespace hand_tracking_mp_lean {
namespace {

using ::testing::HasSubstr;
using ::testing::status::StatusIs;

hand_tracking_mp_lean::ImageFrame GetInputFrame(
    const int width, const int height, const int channel,
    const hand_tracking_mp_lean::ImageFormat::Format image_format) {
  const int total_size = width * height * channel;

  hand_tracking_mp_lean::ImageFrame input_frame(image_format, width, height,
                                    /*alignment_boundary =*/1);
  uint8_t* pixel_data = input_frame.MutablePixelData();
  for (int i = 0; i < total_size; ++i) {
    pixel_data[i] = i % 256;
  }

  return input_frame;
}

hand_tracking_mp_lean::CalculatorGraphConfig::Node GetTestingGraphNode() {
  return ParseTextProtoOrDie<hand_tracking_mp_lean::CalculatorGraphConfig::Node>(
      R"pb(
        calculator: "ScaleImageCalculator"
        input_stream: "input_frames"
        output_stream: "scaled_frames"
        options {
          [mediapipe.ScaleImageCalculatorOptions.ext] {
            input_format: SRGB
            output_format: SRGB
            target_width: 720
            target_height: 720
            preserve_aspect_ratio: true
          }
        }
      )pb");
}

TEST(ScaleImageCalculatorTest, ScaleRegualrSize) {
  auto calculator_node = GetTestingGraphNode();
  hand_tracking_mp_lean::CalculatorRunner runner(calculator_node);

  // Vertical 9:16 720P input frame
  auto input_frame = GetInputFrame(720, 1280, 3, hand_tracking_mp_lean::ImageFormat::SRGB);
  auto input_frame_packet =
      hand_tracking_mp_lean::MakePacket<hand_tracking_mp_lean::ImageFrame>(std::move(input_frame));
  runner.MutableInputs()->Index(0).packets.push_back(
      input_frame_packet.At(hand_tracking_mp_lean::Timestamp(1)));
  MP_ASSERT_OK(runner.Run());
}

TEST(ScaleImageCalculatorTest, ScaleOddSize) {
  auto calculator_node = GetTestingGraphNode();
  hand_tracking_mp_lean::CalculatorRunner runner(calculator_node);

  // 1 x 512 input frame
  auto input_frame = GetInputFrame(1, 512, 3, hand_tracking_mp_lean::ImageFormat::SRGB);
  auto input_frame_packet =
      hand_tracking_mp_lean::MakePacket<hand_tracking_mp_lean::ImageFrame>(std::move(input_frame));
  runner.MutableInputs()->Index(0).packets.push_back(
      input_frame_packet.At(hand_tracking_mp_lean::Timestamp(1)));
  ASSERT_THAT(runner.Run(),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("Image frame is empty before rescaling.")));
}

}  // namespace
}  // namespace hand_tracking_mp_lean
