# adapted from hands_test.py, which looks for a testdata directory that isn't part of the original mediapipe repository ...

import cv2
import numpy as np
from mediapipe.python.solutions import hands as mp_hands


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


def _process_video(model_complexity, video_path,
                   max_num_hands=1,
                   num_landmarks=21,
                   num_dimensions=3):
    # Predict pose landmarks for each frame.
    video_cap = cv2.VideoCapture(video_path)
    landmarks_per_frame = []
    w_landmarks_per_frame = []
    with mp_hands.Hands(
            static_image_mode = False,
            max_num_hands = max_num_hands,
            model_complexity = model_complexity,
            min_detection_confidence = 0.5
            ) as hands:
        while True:
            success, input_frame = video_cap.read()
            if not success:
                break

            input_frame = cv2.cvtColor(input_frame, cv2.COLOR_BGR2RGB)
            frame_shape = input_frame.shape
            result = hands.process(image = input_frame)
            frame_landmarks = np.zeros(
                [max_num_hands,
                 num_landmarks, num_dimensions]
                ) * np.nan
            frame_w_landmarks = np.zeros(
                [max_num_hands,
                 num_landmarks, num_dimensions]
                ) * np.nan

            if result.multi_hand_landmarks:
                for idx, landmarks in enumerate(result.multi_hand_landmarks):
                    landmarks = _landmarks_list_to_array(landmarks, frame_shape)
                    frame_landmarks[idx] = landmarks
            if result.multi_hand_world_landmarks:
                for idx, w_landmarks in enumerate(result.multi_hand_world_landmarks):
                    w_landmarks = _world_landmarks_list_to_array(w_landmarks)
                    frame_w_landmarks[idx] = w_landmarks

            landmarks_per_frame.append(frame_landmarks)
            w_landmarks_per_frame.append(frame_w_landmarks)
    return (np.array(landmarks_per_frame), np.array(w_landmarks_per_frame))


def test_video():
    """ Tests the hand models on a video file. """
    _process_video(model_complexity=1, video_path='video.avi')

if __name__ == '__main__':
    test_video()