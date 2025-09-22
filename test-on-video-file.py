# adapted from hands_test.py, which looks for a testdata directory that isn't part of the original mediapipe repository ...

import cv2
import numpy as np
from mediapipe.python.solutions import hands as mp_hands

num_landmarks=21
num_dimensions=3

def _landmarks_list_to_array(landmark_list, image_shape):
    rows, cols, _ = image_shape
    return np.asarray(
        [(lmk.x * cols, lmk.y * rows, lmk.z * cols)
         for lmk in landmark_list.landmark]
        )


def _world_landmarks_list_to_array(landmark_list):
    return np.asarray(
        [(lmk.x, lmk.y, lmk.z)
         for lmk in landmark_list.landmark]
        )


def _process_video(input_video, model_complexity, max_num_hands=1):
    # Predict pose landmarks for each frame.
    video_cap = cv2.VideoCapture(input_video)
    total_frames = int(video_cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"processing video {input_video} consisting of {total_frames} frames")
    landmarks_per_frame = []
    w_landmarks_per_frame = []
    processed_frames = 0
    frame_idx = 0
    with mp_hands.Hands(
            static_image_mode = False,
            max_num_hands = max_num_hands,
            model_complexity = model_complexity,
            min_detection_confidence = 0.5
            ) as hands:
        while True:
            frame_idx += 1
            success, input_frame = video_cap.read()
            if not success:
                if frame_idx <= total_frames:
                    raise RuntimeError(f"Bad/corrupt frame encountered at frame {frame_idx} (1-based index). Stopping early.")
                break
            processed_frames += 1
            input_frame = cv2.cvtColor(input_frame, cv2.COLOR_BGR2RGB)
            frame_shape = input_frame.shape
            result = hands.process(image = input_frame)

            frame_landmarks = np.empty([max_num_hands, num_landmarks, num_dimensions]) * np.nan
            frame_w_landmarks = np.empty([max_num_hands, num_landmarks, num_dimensions]) * np.nan

            if result.multi_hand_landmarks:
                for idx, landmarks in enumerate(result.multi_hand_landmarks):
                    landmarks = _landmarks_list_to_array(landmarks, frame_shape)
                    frame_landmarks[idx] = landmarks
                # print(frame_landmarks)

            if result.multi_hand_world_landmarks:
                for idx, w_landmarks in enumerate(result.multi_hand_world_landmarks):
                    w_landmarks = _world_landmarks_list_to_array(w_landmarks)
                    frame_w_landmarks[idx] = w_landmarks
                # print(frame_w_landmarks)


    if processed_frames < total_frames:
        raise RuntimeError(f"Video processing stopped early: processed {processed_frames} out of {total_frames} frames. Possible bad/corrupt frame encountered.")


def test_video(input_video):

    """ Tests the hand models on a video file. """

    # The following stderr output is expected at the start of the run:
    # INFO: Created TensorFlow Lite XNNPACK delegate for CPU.
    # WARNING: All log messages before absl::InitializeLog() is called are written to STDERR
    # W0000 00:00:1758560081.963286 2425952 inference_feedback_manager.cc:114] Feedback manager requires a model with a single signature inference. Disabling support for feedback tensors.
    # W0000 00:00:1758560081.975544 2425952 inference_feedback_manager.cc:114] Feedback manager requires a model with a single signature inference. Disabling support for feedback tensors.
    # W0000 00:00:1758560082.720623 2425968 landmark_projection_calculator.cc:78] Using NORM_RECT without IMAGE_DIMENSIONS is only supported for the square ROI. Provide IMAGE_DIMENSIONS or use PROJECTION_MATRIX.

    _process_video(input_video, model_complexity=1)

if __name__ == '__main__':
    import sys
    if not sys.argv[1:]:
        print("Usage: python test-on-video-file.py <input-video>")
        sys.exit(1)
    test_video(input_video=sys.argv[1])