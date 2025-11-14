#ifndef MEDIAPIPE_UTIL_RESOURCES_TEST_UTIL_H_
#define MEDIAPIPE_UTIL_RESOURCES_TEST_UTIL_H_

#include <memory>
#include <string>

#include "absl/container/flat_hash_map.h"
#include "mediapipe/framework/resources.h"

namespace hand_tracking_mp_lean {

// Creates resources which are held solely in memory.
//
// NOTE: Might be useful for testing.
std::unique_ptr<Resources> CreateInMemoryResources(
    absl::flat_hash_map<std::string, std::string> resources);

}  // namespace hand_tracking_mp_lean

#endif  // MEDIAPIPE_UTIL_RESOURCES_TEST_UTIL_H_
