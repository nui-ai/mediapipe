# runs the hands pipeline over an input file, writing its outputs to a protobuf output file.
# adapted from hands_test.py, which looks for a testdata directory that isn't part of the original mediapipe repository ...

import cv2
import numpy as np
import os
from mediapipe.python.solutions import hands as mp_hands
from mediapipe.examples.desktop import pipeline_output_pb2
from mediapipe.framework.formats import landmark_pb2, classification_pb2, rect_pb2, detection_pb2  # only known to the venv where pip install generates them from proto files, not known to the IDE

# Define constant output filename
OUTPUT_FILE_PATH = "output_data_python.pb"

num_landmarks=21
num_dimensions=3

def write_delimited_message(pb_file, message):
    # no python library implementation for this in python https://chatgpt.com/s/t_68de7b5db2bc8191a472dc39f7af2c4a
    # (only in cpp, so we roll our chatgpt own)
    data = message.SerializeToString()
    length = len(data)
    # Write length as protobuf varint
    while True:
        to_write = length & 0x7F
        length >>= 7
        if length:
            pb_file.write(bytes([to_write | 0x80]))
        else:
            pb_file.write(bytes([to_write]))
            break
    pb_file.write(data)

def _process_video(input_video, model_complexity, max_num_hands=2):
    # Predict pose landmarks for each frame.
    video_cap = cv2.VideoCapture(input_video)
    total_frames = int(video_cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Processing video {input_video} consisting of {total_frames} frames")
    landmarks_per_frame = []
    w_landmarks_per_frame = []
    processed_frames = 0
    frame_idx = 0

    # Open file in write binary mode to overwrite any existing file
    with open(OUTPUT_FILE_PATH, "wb") as pb_file:
        with mp_hands.Hands(
                static_image_mode = False,
                max_num_hands = max_num_hands,
                model_complexity = model_complexity,
                min_detection_confidence = 0.5
                ) as hands:
            while True:
                success, input_frame = video_cap.read()
                if not success:
                    if frame_idx < total_frames:
                        raise RuntimeError(f"Bad/corrupt frame encountered at frame {frame_idx} (0-based index). Stopping early.")
                    break

                # put the output into a protobuf message for file writing
                output_msg = pipeline_output_pb2.PipelineOutputData()
                output_msg.frame_number = frame_idx
                processed_frames += 1
                input_frame = cv2.cvtColor(input_frame, cv2.COLOR_BGR2RGB)
                frame_shape = input_frame.shape
                result = hands.process(image = input_frame)

                frame_landmarks = np.empty([max_num_hands, num_landmarks, num_dimensions]) * np.nan
                frame_w_landmarks = np.empty([max_num_hands, num_landmarks, num_dimensions]) * np.nan

                if result.multi_hand_landmarks:
                    for landmarks in result.multi_hand_landmarks:
                        output_msg.multi_hand_landmarks.append(landmarks)

                if result.multi_hand_world_landmarks:
                    for w_landmarks in result.multi_hand_world_landmarks:
                        output_msg.multi_hand_world_landmarks.append(w_landmarks)

                if result.multi_handedness:
                    for handedness in result.multi_handedness:
                        output_msg.multi_handedness.append(handedness)

                # Optionally: hand_rects_from_palm_detections if available
                # if result.hand_rects_from_palm_detections:
                #     for rect in result.hand_rects_from_palm_detections:
                #         output_msg.hand_rects_from_palm_detections.append(rect)

                # Write as delimited protobuf message
                write_delimited_message(pb_file, output_msg)
                frame_idx += 1

    print(f"Writing completed: {processed_frames} frames processed and saved to {os.path.abspath(OUTPUT_FILE_PATH)}")

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
        print("Usage: python hand_tracking_pipeline_run.py <input-video>")
        sys.exit(1)
    test_video(input_video=sys.argv[1])