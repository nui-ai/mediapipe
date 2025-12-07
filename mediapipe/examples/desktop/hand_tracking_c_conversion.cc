#include "mediapipe/examples/desktop/hand_tracking_c_types.h"
#include "mediapipe/examples/desktop/hand_tracking_c_conversion.h"
#include "mediapipe/liberated/hand_tracking.h"
#include <cstdlib>
#include <cstring>

/// deep copy the landmark lists
static void ConvertViewportLandmarksCppToCArray(const std::vector<hand_tracking_mp_lean::NormalizedLandmarkList>* cpp_hand_results, NormalizedLandmarkListC** c_hand_results) {
    size_t n_hands = cpp_hand_results ? cpp_hand_results->size() : 0;
    *c_hand_results = n_hands > 0 ? (NormalizedLandmarkListC*)calloc(n_hands, sizeof(NormalizedLandmarkListC)) : nullptr;
    if (!cpp_hand_results) return;
    for (size_t i = 0; i < n_hands; ++i) {
        const auto& norm_list = (*cpp_hand_results)[i];
        for (int j = 0; j < NUM_LANDMARKS; ++j) { // NUM_LANDMARKS is int
            const auto& lm = norm_list.landmark(j);
            (*c_hand_results)[i].landmark[j].x = lm.x();
            (*c_hand_results)[i].landmark[j].y = lm.y();
            (*c_hand_results)[i].landmark[j].z = lm.z();
            (*c_hand_results)[i].landmark[j].visibility = lm.visibility();
            (*c_hand_results)[i].landmark[j].presence = lm.presence();
        }
    }
}

/// deep copy the landmark lists
static void ConvertObjectLandmarksCppToCArray(const std::vector<hand_tracking_mp_lean::LandmarkList>* cpp_hand_results, LandmarkListC** c_hand_results) {
    size_t n_hands = cpp_hand_results ? cpp_hand_results->size() : 0;
    *c_hand_results = n_hands > 0 ? (LandmarkListC*)calloc(n_hands, sizeof(LandmarkListC)) : nullptr;
    if (!cpp_hand_results) return;
    for (size_t i = 0; i < n_hands; ++i) {
        const auto& obj_list = (*cpp_hand_results)[i];
        for (int j = 0; j < NUM_LANDMARKS; ++j) {
            const auto& lm = obj_list.landmark(j);
            (*c_hand_results)[i].landmark[j].x = lm.x();
            (*c_hand_results)[i].landmark[j].y = lm.y();
            (*c_hand_results)[i].landmark[j].z = lm.z();
            (*c_hand_results)[i].landmark[j].visibility = lm.visibility();
            (*c_hand_results)[i].landmark[j].presence = lm.presence();
        }
    }
}

// deep copy the handedness classification
static void ConvertClassificationsCppToC(const std::vector<hand_tracking_mp_lean::ClassificationList>* cpp_hand_results, ClassificationListC** c_hand_results) {
    size_t n_classes = cpp_hand_results ? cpp_hand_results->size() : 0;
    *c_hand_results = n_classes > 0 ? (ClassificationListC*)calloc(n_classes, sizeof(ClassificationListC)) : nullptr;
    if (!cpp_hand_results) return;
    for (size_t i = 0; i < n_classes; ++i) {
        const auto& cls_list = (*cpp_hand_results)[i];
        size_t n_c = static_cast<size_t>(cls_list.classification_size());
        (*c_hand_results)[i].classification = n_c > 0 ? (ClassificationC*)calloc(n_c, sizeof(ClassificationC)) : nullptr;
        (*c_hand_results)[i].classification_count = n_c;
        for (size_t j = 0; j < n_c; ++j) {
            const auto& c = cls_list.classification(static_cast<int>(j));
            (*c_hand_results)[i].classification[j].index = c.index();
            (*c_hand_results)[i].classification[j].score = c.score();
            (*c_hand_results)[i].classification[j].label = strdup(c.label().c_str());
            (*c_hand_results)[i].classification[j].display_name = strdup(c.display_name().c_str());
        }
    }
}

// deep copy detection details
static void ConvertDetectionDetailsCppToCArray(const std::vector<hand_tracking_mp_lean::DetectionInformation>* cpp_details,
                                               DetectionDetailsC** c_details_out,
                                               size_t* count_out) {
    size_t n = cpp_details ? cpp_details->size() : 0;
    *count_out = n;
    if (n == 0) { *c_details_out = nullptr; return; }
    *c_details_out = (DetectionDetailsC*)calloc(n, sizeof(DetectionDetailsC));
    for (size_t i = 0; i < n; ++i) {
        const auto& d = (*cpp_details)[i];
        (*c_details_out)[i].palm_detection_score = d.palm_detection_score;
        if (d.detected.has_value()) {
            (*c_details_out)[i].has_detected = 1;
            const auto& rv = d.detected.value();
            (*c_details_out)[i].detected.x_center = rv.x_center;
            (*c_details_out)[i].detected.y_center = rv.y_center;
            (*c_details_out)[i].detected.width = rv.width;
            (*c_details_out)[i].detected.height = rv.height;
            (*c_details_out)[i].detected.rotation = rv.rotation;
        } else {
            (*c_details_out)[i].has_detected = 0;
        }
        if (d.oriented.has_value()) {
            (*c_details_out)[i].has_oriented = 1;
            const auto& rv = d.oriented.value();
            (*c_details_out)[i].oriented.x_center = rv.x_center;
            (*c_details_out)[i].oriented.y_center = rv.y_center;
            (*c_details_out)[i].oriented.width = rv.width;
            (*c_details_out)[i].oriented.height = rv.height;
            (*c_details_out)[i].oriented.rotation = rv.rotation;
        } else {
            (*c_details_out)[i].has_oriented = 0;
        }
        if (d.expanded.has_value()) {
            (*c_details_out)[i].has_expanded = 1;
            const auto& rv = d.expanded.value();
            (*c_details_out)[i].expanded.x_center = rv.x_center;
            (*c_details_out)[i].expanded.y_center = rv.y_center;
            (*c_details_out)[i].expanded.width = rv.width;
            (*c_details_out)[i].expanded.height = rv.height;
            (*c_details_out)[i].expanded.rotation = rv.rotation;
        } else {
            (*c_details_out)[i].has_expanded = 0;
        }
    }
}

/// converts the cpp struct to a strict-C struct by copy, while alternatively setting an error message and returning an error indication in case any landmark list isn't the right length.
/// this extra safety is taken so that the C api is robust to use, though the test for the size of the array could be made earlier on and not during this last stage of conversion.
int ConvertCppResultToCNestedStruct(const hand_tracking_mp_lean::ImageHandTrackingResult& cpp_result, HandTrackingResultC* c_result_out, void (*set_last_error)(const std::string& err)) {
    // verify the number of viewport landmarks before passing back as a C array
    const auto* norm_lists = cpp_result.viewport_landmarkss.get();
    c_result_out->viewport_landmarkss_count = norm_lists ? norm_lists->size() : 0;
    if (norm_lists) {
        for (const auto& norm_list : *norm_lists) {
            if (norm_list.landmark_size() != NUM_LANDMARKS) {
                if (set_last_error) set_last_error("NormalizedLandmarkList does not have exactly NUM_LANDMARKS landmarks");
                return -1;
            }
        }
    }

    // verify the number of object landmarks before passing back as a C array
    const auto* obj_lists = cpp_result.object_landmarkss.get();
    c_result_out->object_landmarkss_count = obj_lists ? obj_lists->size() : 0;
    if (obj_lists) {
        for (const auto& obj_list : *obj_lists) {
            if (obj_list.landmark_size() != NUM_LANDMARKS) {
                if (set_last_error) set_last_error("LandmarkList does not have exactly NUM_LANDMARKS landmarks");
                return -1;
            }
        }
    }

    ConvertViewportLandmarksCppToCArray(norm_lists, &c_result_out->viewport_landmarkss);
    ConvertObjectLandmarksCppToCArray(obj_lists, &c_result_out->object_landmarkss);

    // handedness classifications
    const auto* handedness = cpp_result.handedness_classifications.get();
    c_result_out->classificationss_count = handedness ? handedness->size() : 0;
    ConvertClassificationsCppToC(handedness, &c_result_out->classificationss);

    // hand presence scores
    const auto* scores = cpp_result.landmarks_derived_hand_presence_scores.get();
    c_result_out->hand_presence_scores_count = scores ? scores->size() : 0;
    if (scores && !scores->empty()) {
        c_result_out->hand_presence_scores = (float*)calloc(scores->size(), sizeof(float));
        for (size_t i = 0; i < scores->size(); ++i) {
            c_result_out->hand_presence_scores[i] = (*scores)[i];
        }
    } else {
        c_result_out->hand_presence_scores = nullptr;
    }

    // hand rectangles
    const auto* rects = cpp_result.landmarks_based_rectangles.get();
    c_result_out->hand_rects_from_landmarks_count = rects ? rects->size() : 0;
    if (rects && !rects->empty()) {
        c_result_out->hand_rects_from_landmarks = (RectGeometryC*)calloc(rects->size(), sizeof(RectGeometryC));
        for (size_t i = 0; i < rects->size(); ++i) {
            const auto& r = (*rects)[i];
            c_result_out->hand_rects_from_landmarks[i].x_center = r.x_center;
            c_result_out->hand_rects_from_landmarks[i].y_center = r.y_center;
            c_result_out->hand_rects_from_landmarks[i].width = r.width;
            c_result_out->hand_rects_from_landmarks[i].height = r.height;
            c_result_out->hand_rects_from_landmarks[i].rotation = r.rotation;
        }
    } else {
        c_result_out->hand_rects_from_landmarks = nullptr;
    }

    // detection details
    ConvertDetectionDetailsCppToCArray(cpp_result.detections_information.get(), &c_result_out->detections_details, &c_result_out->detection_details_count);

    return 0;
}
