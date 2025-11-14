#include "mediapipe/framework/api2/packet.h"

namespace hand_tracking_mp_lean {
namespace api2 {

PacketBase FromOldPacket(const hand_tracking_mp_lean::Packet& op) {
  return PacketBase(packet_internal::GetHolderShared(op)).At(op.Timestamp());
}

PacketBase FromOldPacket(hand_tracking_mp_lean::Packet&& op) {
  Timestamp t = op.Timestamp();
  return PacketBase(packet_internal::GetHolderShared(std::move(op))).At(t);
}

hand_tracking_mp_lean::Packet ToOldPacket(const PacketBase& p) {
  return hand_tracking_mp_lean::packet_internal::Create(p.payload_, p.timestamp_);
}

hand_tracking_mp_lean::Packet ToOldPacket(PacketBase&& p) {
  return hand_tracking_mp_lean::packet_internal::Create(std::move(p.payload_),
                                            p.timestamp_);
}

}  // namespace api2
}  // namespace hand_tracking_mp_lean
