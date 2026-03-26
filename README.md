Dưới đây là bản **README ngắn gọn và trực quan** cho project của em, tập trung vào ý chính: dataset, model, và cách chạy.

---

# Sign Language Recognition

## Mô tả dự án

Dự án này nhận diện **100 từ cơ bản bằng ngôn ngữ ký hiệu Mỹ (ASL)** từ video, sử dụng **MediaPipe Hands** để trích xuất keypoints bàn tay và **LSTM** để dự đoán từ.

---

## Yêu cầu

* Python ≥ 3.8
* OpenCV
* MediaPipe
* TensorFlow / Keras
* NumPy, scikit-learn, tqdm

Cài đặt nhanh:

```bash
pip install opencv-python mediapipe tensorflow numpy scikit-learn tqdm
```

---

## Cấu trúc thư mục

```
project2/
├─ dataset/
│  ├─ videos/           # Video gốc
│  ├─ keypoints/        # File .npy lưu keypoints
│  ├─ nslt_100.json     # Metadata video
│  └─ wlasl_class_list.txt # Mapping class ID -> từ
├─ sign_language_model.h5 # Model đã huấn luyện
└─ main.py              # Code chính
```

---

## Cách sử dụng

### 1. Trích xuất keypoints từ video

```python
# Pad sequence về 120 frames và lưu file .npy
from pathlib import Path
import numpy as np
seq_padded = pad_sequence(sequence)
np.save(KEYPOINT_PATH / "video_id.npy", seq_padded)
```

### 2. Huấn luyện model

```python
model.fit(X_train, y_train, epochs=50, batch_size=8, validation_data=(X_test, y_test))
model.save("sign_language_model.h5")
```

### 3. Dự đoán real-time từ webcam

```python
# Khởi tạo MediaPipe Hands và load model
seq = []  # Sliding window
# Dự đoán khi đủ 120 frames
res = model.predict(np.expand_dims(seq, axis=0))
pred_label = np.argmax(res)
```

### 4. Trình phát video dataset (có vietsub)

* Bấm `n` để chuyển video
* Bấm `q` để thoát

---

## Lưu ý

* Dataset chỉ xử lý **100 từ đầu tiên** để giảm thời gian huấn luyện.
* File `.npy` của video thiếu sẽ gây lỗi khi dự đoán, nên **bắt buộc tạo keypoints trước**.
* Model dự đoán ra **Class ID**, có thể map sang chữ bằng `wlasl_class_list.txt`.


