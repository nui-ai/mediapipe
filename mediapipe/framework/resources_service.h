#ifndef MEDIAPIPE_FRAMEWORK_RESOURCES_SERVICE_H_
#define MEDIAPIPE_FRAMEWORK_RESOURCES_SERVICE_H_

#include "mediapipe/framework/graph_service.h"
#include "mediapipe/framework/resources.h"

namespace hand_tracking_mp_lean {

// The service represents resources (files, assets, etc.) subgraph and
// calculators can load using `SubgraphContext::GetResources` and
// `CalculatorContext::GetResources` respectively.
//
// NOTE: calculators don't need to call `UseService` for this particular
// service.
inline constexpr GraphService<Resources> kResourcesService(
    "kResourcesService", GraphServiceBase::kDisallowDefaultInitialization);

}  // namespace hand_tracking_mp_lean

#endif  // MEDIAPIPE_FRAMEWORK_RESOURCES_SERVICE_H_
