#include "mediapipe/examples/desktop/hand_tracking_c_types.h"
#include <stdlib.h>

extern "C" void hand_tracking_result_destroy(HandTrackingResultC* result) {
    if (!result) return;
    // Free normalized landmarks
    if (result->normalized_landmarkss) {
        free(result->normalized_landmarkss);
    }
    // Free object landmarks
    if (result->landmarkss) {
        free(result->landmarkss);
    }
    // Free classifications
    if (result->classificationss) {
        // Note: we only have a single ClassificationListC here (per-hand array would require a count).
        // Free nested strings in that one list, if allocated.
        if (result->classificationss->classification && result->classificationss->classification_count > 0) {
            for (size_t i = 0; i < result->classificationss->classification_count; ++i) {
                ClassificationC* c = &result->classificationss->classification[i];
                if (c->label) free((void*)c->label);
                if (c->display_name) free((void*)c->display_name);
            }
            free(result->classificationss->classification);
        }
        free(result->classificationss);
    }
    // Free detection details
    if (result->detection_details) {
        free(result->detection_details);
    }
    free(result);
}

