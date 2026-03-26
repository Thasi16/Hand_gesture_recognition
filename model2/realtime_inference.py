import cv2
import numpy as np
import mediapipe as mp
from tensorflow.keras.models import load_model
from config import DICT_PATH, MAX_FRAMES, MODEL_PATH
from utils import extract_keypoints

def load_dictionary():
    class_mapping = {}
    try:
        with open(DICT_PATH, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 2:
                    class_mapping[int(parts[0])] = " ".join(parts[1:]).upper()
    except FileNotFoundError:
        print("⚠️ Không tìm thấy file từ điển, sẽ hiển thị Class ID.")
    return class_mapping

def main():
    model = load_model(MODEL_PATH)
    class_mapping = load_dictionary()

    sequence = []
    current_word = "..."
    confidence = 0.0

    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    cap = cv2.VideoCapture(0)

    with mp_hands.Hands(model_complexity=0, min_detection_confidence=0.5, min_tracking_confidence=0.5) as hands:
        print("AI READY! Hãy múa trước camera (Bấm 'q' để thoát)")

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            
            frame = cv2.flip(frame, 1)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(frame_rgb)
            
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Trích xuất keypoints (Dùng chung hàm chuẩn hóa với file Train)
            keypoints = extract_keypoints(results)
            sequence.append(keypoints)
            sequence = sequence[-MAX_FRAMES:]
            
            if len(sequence) == MAX_FRAMES:
                input_data = np.expand_dims(sequence, axis=0)
                res = model.predict(input_data, verbose=0)[0]
                idx = np.argmax(res)
                confidence = res[idx]
                
                if confidence > 0.5: # Đã nâng ngưỡng tin cậy lên 50% để tránh nhiễu
                    current_word = class_mapping.get(idx, f"Class {idx}")

            # Vẽ HUD
            cv2.rectangle(frame, (0, 0), (640, 60), (245, 117, 16), -1)
            display_text = f"{current_word} ({confidence*100:.1f}%)"
            cv2.putText(frame, display_text, (15, 45), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2, cv2.LINE_AA)
            
            cv2.imshow('WLASL Recognition', frame)
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
