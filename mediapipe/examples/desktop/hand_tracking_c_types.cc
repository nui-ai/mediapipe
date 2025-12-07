#include "mediapipe/examples/desktop/hand_tracking_c_types.h"
#include <cstdlib>

extern "C" void hand_tracking_result_destroy(HandTrackingResultC* result) {
    if (!result) return;
    // Free normalized landmarks
    if (result->viewport_landmarkss) {
        free(result->viewport_landmarkss);
    }
    // Free object landmarks
    if (result->object_landmarkss) {
        free(result->object_landmarkss);
    }
    // Free classifications
    if (result->classificationss) {
        for (size_t i = 0; i < result->classificationss_count; ++i) {
            ClassificationListC* clist = &result->classificationss[i];
            if (clist->classification && clist->classification_count > 0) {
                for (size_t j = 0; j < clist->classification_count; ++j) {
                    ClassificationC* c = &clist->classification[j];
                    if (c->label) free((void*)c->label);
                    if (c->display_name) free((void*)c->display_name);
                }
                free(clist->classification);
            }
        }
        free(result->classificationss);
    }
    // Free hand presence scores
    if (result->hand_presence_scores) {
        free(result->hand_presence_scores);
    }
    // Free hand rectangles
    if (result->hand_rects_from_landmarks) {
        free(result->hand_rects_from_landmarks);
    }
    // Free detection details
    if (result->detections_details) {
        free(result->detections_details);
    }
    free(result);
}
