#include "mediapipe/examples/desktop/hand_tracking_c_types.h"
#include "mediapipe/examples/desktop/hand_tracking_c_conversion.h"
#include "mediapipe/liberated/liberated_core.h"
#include <stdlib.h>
#include <string.h>

/// deep copy the landmark lists
static void ConvertViewportLandmarksCppToCArray(const std::vector<hand_tracking_mp_lean::NormalizedLandmarkList>* cpp_hand_results, NormalizedLandmarkListC** c_hand_results) {
    size_t n_hands = cpp_hand_results ? cpp_hand_results->size() : 0;
    *c_hand_results = (NormalizedLandmarkListC*)calloc(n_hands, sizeof(NormalizedLandmarkListC));
    for (size_t i = 0; i < n_hands; ++i) {
        const auto& norm_list = (*cpp_hand_results)[i];
        for (size_t j = 0; j < NUM_LANDMARKS; ++j) {
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
    *c_hand_results = (LandmarkListC*)calloc(n_hands, sizeof(LandmarkListC));
    for (size_t i = 0; i < n_hands; ++i) {
        const auto& obj_list = (*cpp_hand_results)[i];
        for (size_t j = 0; j < NUM_LANDMARKS; ++j) {
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
    *c_hand_results = (ClassificationListC*)calloc(n_classes, sizeof(ClassificationListC));
    for (size_t i = 0; i < n_classes; ++i) {
        const auto& cls_list = (*cpp_hand_results)[i];
        size_t n_c = cls_list.classification_size();
        (*c_hand_results)[i].classification = (ClassificationC*)calloc(n_c, sizeof(ClassificationC));
        (*c_hand_results)[i].classification_count = n_c;
        for (size_t j = 0; j < n_c; ++j) {
            const auto& c = cls_list.classification(j);
            (*c_hand_results)[i].classification[j].index = c.index();
            (*c_hand_results)[i].classification[j].score = c.score();
            (*c_hand_results)[i].classification[j].label = strdup(c.label().c_str());
            (*c_hand_results)[i].classification[j].display_name = strdup(c.display_name().c_str());
        }
    }
}

/// converts the cpp struct to a strict-C struct by copy, while alternatively setting an error message and returning an error indication in case any landmark list isn't the right length.
/// this extra safety is taken so that the C api is robust to use, though the test for the size of the array could be made earlier on and not during this last stage of conversion.
int ConvertCppResultToCNestedStruct(const hand_tracking_mp_lean::ImageHandTrackingAndInferenceResult& cpp_result, HandTrackingResultC* c_result_out, void (*set_last_error)(const std::string& err)) {

    // verify the number of viewport landmarks before passing back as a C array
    const auto* norm_lists = cpp_result.viewport_landmarkss.get();
    if (norm_lists) {
        for (size_t i = 0; i < norm_lists->size(); ++i) {
            if ((*norm_lists)[i].landmark_size() != NUM_LANDMARKS) {
                if (set_last_error) set_last_error("NormalizedLandmarkList does not have exactly NUM_LANDMARKS landmarks");
                return -1;
            }
        }
    }

    // verify the number of object landmarks before passing back as a C array
    const auto* obj_lists = cpp_result.object_landmarkss.get();
    if (obj_lists) {
        for (size_t i = 0; i < obj_lists->size(); ++i) {
            if ((*obj_lists)[i].landmark_size() != NUM_LANDMARKS) {
                if (set_last_error) set_last_error("LandmarkList does not have exactly NUM_LANDMARKS landmarks");
                return -1;
            }
        }
    }

    ConvertViewportLandmarksCppToCArray(norm_lists, &c_result_out->normalized_landmarkss);
    ConvertObjectLandmarksCppToCArray(obj_lists, &c_result_out->landmarkss);
    ConvertClassificationsCppToC(cpp_result.handedness_classifications.get(), &c_result_out->classificationss);
    return 0;
}
