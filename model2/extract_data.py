import cv2
import json
import numpy as np
import mediapipe as mp
from tqdm import tqdm
from config import VIDEO_PATH, JSON_PATH, KEYPOINT_PATH, MAX_FRAMES
from utils import extract_keypoints, pad_sequence

def main():
    with open(JSON_PATH, "r") as f:
        metadata = json.load(f)

    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=0.5
    )

    X_list, y_list = [], []

    for video_id, info in tqdm(metadata.items()):
        label = info["action"][0]
        if label >= 10: # Chỉ lấy 10 class đầu tiên
            continue
            
        video_file = VIDEO_PATH / f"{video_id}.mp4"
        if not video_file.exists():
            continue

        cap = cv2.VideoCapture(str(video_file))
        sequence = []

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_rgb.flags.writeable = False                  
            results = hands.process(frame_rgb)
            
            keypoints = extract_keypoints(results)
            sequence.append(keypoints)

        cap.release()
        
        if len(sequence) > 0:
            seq_padded = pad_sequence(sequence, MAX_FRAMES)
            # Lưu riêng lẻ để dự phòng
            np.save(KEYPOINT_PATH / f"{video_id}.npy", seq_padded)
            X_list.append(seq_padded)
            y_list.append(label)

    # Lưu bộ dataset hoàn chỉnh để train
    X = np.array(X_list)
    y = np.array(y_list)
    np.save("X_data.npy", X)
    np.save("y_data.npy", y)
    
    print(f"\n Xong! Đã lưu file X_data.npy {X.shape} và y_data.npy {y.shape}")

if __name__ == "__main__":
    main()
