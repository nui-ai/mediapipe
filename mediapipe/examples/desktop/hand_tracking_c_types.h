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

// Top-level result struct
typedef struct {
    NormalizedLandmarkListC* normalized_landmarks;
    LandmarkListC* landmarks;
    ClassificationListC* classifications;
} HandTrackingResultC;

#ifdef __cplusplus
}
#endif

#endif // HAND_TRACKING_C_TYPES_H_
