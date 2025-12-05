#include "mediapipe/examples/desktop/pipeline_output_util.h"

namespace hand_tracking_mp_lean {

void FillPipelineOutputDataFromCResult(const HandTrackingResultC* c_result, PipelineOutputData* proto, int frame_number) {
    proto->set_frame_number(frame_number);
    // Landmarks
    if (c_result->landmarkss) {
        for (size_t i = 0; c_result->landmarkss[i].landmark[0].x != 0.0f || c_result->landmarkss[i].landmark[0].y != 0.0f || i == 0; ++i) {
            auto* l = proto->add_multi_hand_world_landmarks();
            for (int j = 0; j < 21; ++j) {
                auto* lm = l->add_landmark();
                lm->set_x(c_result->landmarkss[i].landmark[j].x);
                lm->set_y(c_result->landmarkss[i].landmark[j].y);
                lm->set_z(c_result->landmarkss[i].landmark[j].z);
                lm->set_visibility(c_result->landmarkss[i].landmark[j].visibility);
                lm->set_presence(c_result->landmarkss[i].landmark[j].presence);
            }
        }
    }
    if (c_result->normalized_landmarkss) {
        for (size_t i = 0; c_result->normalized_landmarkss[i].landmark[0].x != 0.0f || c_result->normalized_landmarkss[i].landmark[0].y != 0.0f || i == 0; ++i) {
            auto* l = proto->add_multi_hand_landmarks();
            for (int j = 0; j < 21; ++j) {
                auto* lm = l->add_landmark();
                lm->set_x(c_result->normalized_landmarkss[i].landmark[j].x);
                lm->set_y(c_result->normalized_landmarkss[i].landmark[j].y);
                lm->set_z(c_result->normalized_landmarkss[i].landmark[j].z);
                lm->set_visibility(c_result->normalized_landmarkss[i].landmark[j].visibility);
                lm->set_presence(c_result->normalized_landmarkss[i].landmark[j].presence);
            }
        }
    }
    if (c_result->classificationss) {
        // Not implemented: fill multi_handedness if needed
    }
    if (c_result->detection_details) {
        for (size_t i = 0; i < c_result->detection_details_count; ++i) {
            auto* r = proto->add_hand_rects_from_palm_detections();
            r->set_x_center(c_result->detection_details[i].expanded.x_center);
            r->set_y_center(c_result->detection_details[i].expanded.y_center);
            r->set_width(c_result->detection_details[i].expanded.width);
            r->set_height(c_result->detection_details[i].expanded.height);
            r->set_rotation(c_result->detection_details[i].expanded.rotation);
        }
    }
}

} // namespace hand_tracking_mp_lean

