// clone of ImagePropertiesCalculator

#include "mediapipe/framework/api2/node.h"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/formats/image.h"
#include "mediapipe/framework/formats/image_frame.h"

#if !MEDIAPIPE_DISABLE_GPU
#include "mediapipe/gpu/gpu_buffer.h"
#endif  // !MEDIAPIPE_DISABLE_GPU

namespace hand_tracking_mp_lean {
namespace api2 {

#if MEDIAPIPE_DISABLE_GPU
// Just a placeholder to not have to depend on hand_tracking_mp_lean::GpuBuffer.
using GpuBuffer = AnyType;
#else
using GpuBuffer = hand_tracking_mp_lean::GpuBuffer;
#endif  // MEDIAPIPE_DISABLE_GPU

// Extracts image properties from the input image and outputs the properties.
// Currently only supports image size.
// Input:
//   One of the following:
//   IMAGE: An Image or ImageFrame (for backward compatibility with existing
//          graphs that use IMAGE for ImageFrame input)
//   IMAGE_CPU: An ImageFrame
//   IMAGE_GPU: A GpuBuffer
//
// Output:
//   SIZE: Size (as a std::pair<int, int>) of the input image.
//
// Example usage:
// node {
//   calculator: "ImagePropertiesCalculator"
//   input_stream: "IMAGE:image"
//   output_stream: "SIZE:size"
// }
class ImageSize2 : public Node {
 public:
  static constexpr Input<
      OneOf<hand_tracking_mp_lean::Image, hand_tracking_mp_lean::ImageFrame>>::Optional kIn{"IMAGE"};
  // IMAGE_CPU, dedicated to ImageFrame input, is only needed in some top-level
  // graphs for the Python Solution APIs to figure out the type of input stream
  // without running into ambiguities from IMAGE.
  // TODO: Remove IMAGE_CPU once Python Solution APIs adopt Image.
  static constexpr Input<hand_tracking_mp_lean::ImageFrame>::Optional kInCpu{"IMAGE_CPU"};
  static constexpr Input<GpuBuffer>::Optional kInGpu{"IMAGE_GPU"};
  static constexpr Output<std::pair<int, int>> kOut{"SIZE"};

  MEDIAPIPE_NODE_CONTRACT(kIn, kInCpu, kInGpu, kOut);

  static absl::Status UpdateContract(CalculatorContract* cc) {
    RET_CHECK_EQ(kIn(cc).IsConnected() + kInCpu(cc).IsConnected() +
                     kInGpu(cc).IsConnected(),
                 1)
        << "One and only one of IMAGE, IMAGE_CPU and IMAGE_GPU input is "
           "expected.";

    return absl::OkStatus();
  }

  absl::Status Process(CalculatorContext* cc) override {
    std::pair<int, int> size;

    if (kIn(cc).IsConnected()) {
      kIn(cc).Visit(
          [&size](const hand_tracking_mp_lean::Image& value) {
            size.first = value.width();
            size.second = value.height();
          },
          [&size](const hand_tracking_mp_lean::ImageFrame& value) {
            size.first = value.Width();
            size.second = value.Height();
          });
    }
    if (kInCpu(cc).IsConnected()) {
      const auto& image = *kInCpu(cc);
      size.first = image.Width();
      size.second = image.Height();
    }
#if !MEDIAPIPE_DISABLE_GPU
    if (kInGpu(cc).IsConnected()) {
      const auto& image = *kInGpu(cc);
      size.first = image.width();
      size.second = image.height();
    }
#endif  // !MEDIAPIPE_DISABLE_GPU

    kOut(cc).Send(size);

    return absl::OkStatus();
  }
};

MEDIAPIPE_REGISTER_NODE(ImageSize2);

}  // namespace api2
}  // namespace hand_tracking_mp_lean
