// Copyright 2019 The MediaPipe Authors.
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

#include <functional>
#include <map>
#include <memory>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include "absl/log/absl_check.h"
#include "absl/strings/str_cat.h"
#include "mediapipe/framework/calculator.pb.h"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/calculator_options.pb.h"
#include "mediapipe/framework/port/canonical_errors.h"
#include "mediapipe/framework/port/ret_check.h"
#include "mediapipe/framework/port/status.h"
#include "mediapipe/framework/tool/container_util.h"
#include "mediapipe/framework/tool/name_util.h"
#include "mediapipe/framework/tool/subgraph_expansion.h"
#include "mediapipe/framework/tool/switch_container.pb.h"

namespace mediapipe {
    namespace tool {
        using mediapipe::SwitchContainerOptions;

        // A graph factory producing a CalculatorGraphConfig routing packets to
        // one of several contained CalculatorGraphConfigs.
        //
        // Usage example:
        //
        //     node {
        //       calculator: "SwitchContainer"
        //       input_stream: "ENABLE:enable"
        //       input_stream: "INPUT_VIDEO:video_frames"
        //       output_stream: "OUTPUT_VIDEO:output_frames"
        //       options {
        //         [mediapipe.SwitchContainerOptions.ext] {
        //           contained_node: { calculator: "BasicSubgraph" }
        //           contained_node: { calculator: "AdvancedSubgraph" }
        //         }
        //       }
        //     }
        //
        // Note that the input and output stream tags supplied to the container node
        // must match the input and output stream tags required by the contained nodes,
        // such as "INPUT_VIDEO" and "OUTPUT_VIDEO" in the example above.
        //
        // Input stream "ENABLE" specifies routing of packets to either contained_node 0
        // or contained_node 1, given "ENABLE:false" or "ENABLE:true" respectively.
        // Input-side-packet "ENABLE" and input-stream "SELECT" can also be used
        // similarly to specify the active channel.
        //
        // Note that this container defaults to use ImmediateInputStreamHandler,
        // which can be used to accept infrequent "enable" packets asynchronously.
        // However, it can be overridden to work with DefaultInputStreamHandler,
        // which can be used to accept frequent "enable" packets synchronously.
        class SwitchContainer : public Subgraph {
        public:
            SwitchContainer() = default;
            absl::StatusOr<CalculatorGraphConfig> GetConfig(
                const Subgraph::SubgraphOptions& options) override;
        };
        REGISTER_MEDIAPIPE_GRAPH(SwitchContainer);
    }
}