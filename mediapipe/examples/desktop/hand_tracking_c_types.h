// Pure C types for hand tracking results
#ifndef HAND_TRACKING_C_TYPES_H_
#define HAND_TRACKING_C_TYPES_H_

#include <cstddef>
#include <cstdint>

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
} RectGeometryC;

typedef struct {
    float palm_detection_score;   // SSD palm detection score
    // Each rectangle is optional; a corresponding has_* flag indicates presence (1) or absence (0)
    uint8_t has_detected;
    RectGeometryC detected; // initial axis-aligned palm detection
    uint8_t has_oriented;
    RectGeometryC oriented; // oriented detection rect (pre-expansion)
    uint8_t has_expanded;
    RectGeometryC expanded; // values after expansion
} DetectionDetailsC;

// Top-level result struct
typedef struct HandTrackingResultC {

    NormalizedLandmarkListC* viewport_landmarkss; // array, one per hand
    size_t viewport_landmarkss_count;

    LandmarkListC* object_landmarkss; // array, one per hand
    size_t object_landmarkss_count;

    ClassificationListC* classificationss; // array, one per hand
    size_t classificationss_count;

    float* hand_presence_scores; // array, one per hand
    size_t hand_presence_scores_count;

    RectGeometryC* hand_rects_from_landmarks; // array, one per hand
    size_t hand_rects_from_landmarks_count;

    DetectionDetailsC* detections_details; // array, one per detection
    size_t detection_details_count;
} HandTrackingResultC;

#ifdef __cplusplus
}
#endif

#endif // HAND_TRACKING_C_TYPES_H_
