import numpy as np

def extract_keypoints(results):
    """Trích xuất và chuẩn hóa tọa độ 126 điểm của 2 bàn tay"""
    keypoints = np.zeros(126)
    if results.multi_hand_landmarks:
        coords = []
        for hand_landmarks in results.multi_hand_landmarks:
            wrist = hand_landmarks.landmark[0]
            # Tính khoảng cách từ cổ tay đến gốc ngón giữa để làm thước đo (scale)
            m_joint = hand_landmarks.landmark[9]
            scale = np.sqrt((m_joint.x - wrist.x)**2 + (m_joint.y - wrist.y)**2) + 1e-6

            for lm in hand_landmarks.landmark:
                # Trừ gốc cổ tay và chia cho scale để cố định kích thước bàn tay
                coords.extend([
                    (lm.x - wrist.x) / scale, 
                    (lm.y - wrist.y) / scale, 
                    (lm.z - wrist.z) / scale
                ])
        coords = coords[:126]
        keypoints[:len(coords)] = coords
    return keypoints

def pad_sequence(seq, max_frames):
    """Đệm video về đúng số frame quy định"""
    if len(seq) > max_frames:
        return seq[:max_frames]
    padding = np.zeros((max_frames - len(seq), 126))
    return np.vstack((seq, padding))
