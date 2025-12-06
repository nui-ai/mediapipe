// Pure C types for hand tracking results
#ifndef HAND_TRACKING_C_TYPES_H_
#define HAND_TRACKING_C_TYPES_H_

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#ifndef NUM_LANDMARKS
#define NUM_LANDMARKS 21
#endif

// C struct for a single landmark
typedef struct {
    float x;
    float y;
    float z;
    float visibility;
    float presence;
} LandmarkC;

// C struct for a list of landmarks
typedef struct {
    LandmarkC landmark[NUM_LANDMARKS];
} LandmarkListC;

// C struct for a single normalized landmark
typedef struct {
    float x;
    float y;
    float z;
    float visibility;
    float presence;
} NormalizedLandmarkC;

// C struct for a list of normalized landmarks
typedef struct {
    NormalizedLandmarkC landmark[NUM_LANDMARKS];
} NormalizedLandmarkListC;

// C struct for a single classification
typedef struct {
    int32_t index;
    float score;
    const char* label;
    const char* display_name;
} ClassificationC;

// C struct for a list of classifications
typedef struct {
    ClassificationC* classification;
    size_t classification_count;
} ClassificationListC;

// Details for detection debug/analysis. Image-normalized rectangle:
// x,y are center in [0,1] left->right, top->bottom; width/height in [0,1]; rotation in radians CCW.
typedef struct {
    float x_center;
    float y_center;
    float width;
    float height;
    float rotation;
} RectValuesC;

typedef struct {
    float palm_detection_score;   // SSD palm detection score
    float hand_presence_score;    // landmarks-inference presence score
    // Each rectangle is optional; a corresponding has_* flag indicates presence (1) or absence (0)
    uint8_t has_detected;
    RectValuesC detected; // initial axis-aligned palm detection
    uint8_t has_oriented;
    RectValuesC oriented; // oriented detection rect (pre-expansion)
    uint8_t has_expanded;
    RectValuesC expanded; // values after expansion
    uint8_t has_hand_rect_for_next_frame;
    RectValuesC hand_rect_for_next_frame; // predicted rect for next frame after expansion
} DetectionDetailsC;

// Top-level result struct
typedef struct {
    NormalizedLandmarkListC* normalized_landmarkss;
    LandmarkListC* landmarkss;
    ClassificationListC* classificationss;
    DetectionDetailsC* detection_details; // per-detection raw/expanded details
    size_t detection_details_count;
} HandTrackingResultC;

#ifdef __cplusplus
}
#endif

#endif // HAND_TRACKING_C_TYPES_H_
