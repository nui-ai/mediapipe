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
        for (size_t i = 0; i < result->classificationss->classification_count; ++i) {
            ClassificationC* c = &result->classificationss->classification[i];
            if (c->label) free((void*)c->label);
            if (c->display_name) free((void*)c->display_name);
        }
        free(result->classificationss->classification);
        free(result->classificationss);
    }
    free(result);
}

