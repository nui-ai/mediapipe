#include "mediapipe/examples/desktop/hand_tracking_c_types.h"
#include <stdlib.h>

extern "C" void hand_tracking_result_destroy(HandTrackingResultC* result) {
    if (!result) return;
    // Free normalized landmarks
    if (result->normalized_landmarks) {
        free(result->normalized_landmarks);
    }
    // Free object landmarks
    if (result->landmarks) {
        free(result->landmarks);
    }
    // Free classifications
    if (result->classifications) {
        for (size_t i = 0; i < result->classifications->classification_count; ++i) {
            ClassificationC* c = &result->classifications->classification[i];
            if (c->label) free((void*)c->label);
            if (c->display_name) free((void*)c->display_name);
        }
        free(result->classifications->classification);
        free(result->classifications);
    }
    free(result);
}

