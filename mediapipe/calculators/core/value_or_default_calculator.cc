#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/port/status.h"

namespace hand_tracking_mp_lean {
namespace {

constexpr char kInputValueTag[] = "IN";
constexpr char kTickerTag[] = "TICK";
constexpr char kOutputTag[] = "OUT";
constexpr char kIndicationTag[] = "FLAG";

}  // namespace
// For every packet received on the TICK stream, if the IN stream is not
// empty - emit its value as is as OUT. Otherwise output a default packet.
// FLAG outputs true every time the default value has been used. It does not
//   output anything when IN has a value.
//
// Example config:
// node {
//   calculator: "ValueOrDefaultCalculator"
//   input_stream: "IN:sometimes_missing_value"
//   input_stream: "TICK:clock"
//   output_stream: "OUT:value_or_default"
//   output_stream: "FLAG:used_default"
//   input_side_packet: "default"
// }
//
// TODO: Consider adding an option for a default value as a input-stream
// instead of a side-packet, so it will enable using standard calculators
// instead of creating a new packet-generators. It will also allow a dynamic
// default value.
class ValueOrDefaultCalculator : public hand_tracking_mp_lean::CalculatorBase {
 public:
  ValueOrDefaultCalculator() {}

  ValueOrDefaultCalculator(const ValueOrDefaultCalculator&) = delete;
  ValueOrDefaultCalculator& operator=(const ValueOrDefaultCalculator&) = delete;

  static hand_tracking_mp_lean::Status GetContract(hand_tracking_mp_lean::CalculatorContract* cc) {
    cc->Inputs().Tag(kInputValueTag).SetAny();
    cc->Inputs().Tag(kTickerTag).SetAny();
    cc->Outputs().Tag(kOutputTag).SetSameAs(&cc->Inputs().Tag(kInputValueTag));
    cc->Outputs().Tag(kIndicationTag).Set<bool>();
    cc->InputSidePackets().Index(0).SetSameAs(
        &cc->Inputs().Tag(kInputValueTag));

    return hand_tracking_mp_lean::OkStatus();
  }

  hand_tracking_mp_lean::Status Open(hand_tracking_mp_lean::CalculatorContext* cc) override {
    if (!cc->Inputs().Tag(kInputValueTag).Header().IsEmpty()) {
      cc->Outputs()
          .Tag(kOutputTag)
          .SetHeader(cc->Inputs().Tag(kInputValueTag).Header());
    }
    default_ = cc->InputSidePackets().Index(0);
    cc->SetOffset(hand_tracking_mp_lean::TimestampDiff(0));
    return hand_tracking_mp_lean::OkStatus();
  }

  hand_tracking_mp_lean::Status Process(hand_tracking_mp_lean::CalculatorContext* cc) override {
    // Output according to the TICK signal.
    if (cc->Inputs().Tag(kTickerTag).IsEmpty()) {
      return hand_tracking_mp_lean::OkStatus();
    }
    if (!cc->Inputs().Tag(kInputValueTag).IsEmpty()) {
      // Output the input as is:
      cc->Outputs()
          .Tag(kOutputTag)
          .AddPacket(cc->Inputs().Tag(kInputValueTag).Value());
    } else {
      // Output default:
      cc->Outputs()
          .Tag(kOutputTag)
          .AddPacket(default_.At(cc->InputTimestamp()));
      cc->Outputs()
          .Tag(kIndicationTag)
          .Add(new bool(true), cc->InputTimestamp());
    }
    return hand_tracking_mp_lean::OkStatus();
  }

 private:
  // The default value to replicate every time there is no new value.
  hand_tracking_mp_lean::Packet default_;
};

REGISTER_CALCULATOR(ValueOrDefaultCalculator);

}  // namespace hand_tracking_mp_lean
