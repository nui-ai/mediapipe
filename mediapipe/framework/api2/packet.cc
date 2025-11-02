#include "mediapipe/framework/api2/packet.h"

namespace mediapipe_v01013_based {
namespace api2 {

PacketBase FromOldPacket(const mediapipe_v01013_based::Packet& op) {
  return PacketBase(packet_internal::GetHolderShared(op)).At(op.Timestamp());
}

PacketBase FromOldPacket(mediapipe_v01013_based::Packet&& op) {
  Timestamp t = op.Timestamp();
  return PacketBase(packet_internal::GetHolderShared(std::move(op))).At(t);
}

mediapipe_v01013_based::Packet ToOldPacket(const PacketBase& p) {
  return mediapipe_v01013_based::packet_internal::Create(p.payload_, p.timestamp_);
}

mediapipe_v01013_based::Packet ToOldPacket(PacketBase&& p) {
  return mediapipe_v01013_based::packet_internal::Create(std::move(p.payload_),
                                            p.timestamp_);
}

}  // namespace api2
}  // namespace mediapipe_v01013_based
