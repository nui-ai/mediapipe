# ...existing code...
import cv2
import mediapipe as mp
import json
import os

mp_hands = mp.solutions.hands

# Path to input video file (assumed to be next to this script)
video_path = os.path.join(os.path.dirname(__file__), 'input.avi')
output_path = os.path.join(os.path.dirname(__file__), 'verified-detections.json')

cap = cv2.VideoCapture(video_path)
results_list = []

with mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as hands:
    frame_idx = 0
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            break
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        result = hands.process(image_rgb)
        frame_data = {'frame': frame_idx, 'hands': []}
        if result.multi_hand_landmarks:
            for hand_landmarks, handedness in zip(result.multi_hand_landmarks, result.multi_handedness):
                hand_dict = {
                    'handedness': handedness.classification[0].label,
                    'score': handedness.classification[0].score,
                    'landmarks': [
                        {'x': lm.x, 'y': lm.y, 'z': lm.z}
                        for lm in hand_landmarks.landmark
                    ]
                }
                frame_data['hands'].append(hand_dict)
        results_list.append(frame_data)
        frame_idx += 1

cap.release()

with open(output_path, 'w') as f:
    json.dump(results_list, f, indent=2)

print(f"Detections written to {output_path}")
# ...existing code...

