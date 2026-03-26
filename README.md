```markdown
#  Nhận Diện Ngôn Ngữ Ký Hiệu Thời Gian Thực (Real-time Sign Language Recognition)

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange)
![OpenCV](https://img.shields.io/badge/OpenCV-4.x-green)
![MediaPipe](https://img.shields.io/badge/MediaPipe-Latest-yellow)

Dự án này ứng dụng **Computer Vision** (Thị giác máy tính) và **Deep Learning** (Học sâu) để nhận diện Ngôn ngữ Ký hiệu Mỹ (ASL) theo thời gian thực thông qua Webcam. Mô hình được huấn luyện trên bộ dữ liệu **WLASL** (Word-Level American Sign Language) nổi tiếng, có khả năng dịch các chuyển động tay thành văn bản tiếng Anh trực tiếp.

---

##  Mục lục
1. [Giới thiệu dự án](#-giới-thiệu-dự-án)
2. [Tính năng nổi bật](#-tính-năng-nổi-bật)
3. [Kiến trúc hệ thống](#-kiến-trúc-hệ-thống)
4. [Bộ dữ liệu (Dataset)](#-bộ-dữ-liệu-dataset)
5. [Cấu trúc thư mục](#-cấu-trúc-thư-mục)
6. [Hướng dẫn cài đặt](#-hướng-dẫn-cài-đặt)
7. [Cách sử dụng](#-cách-sử-dụng)
8. [Tùy chỉnh](#-tùy-chỉnh)
9. [Hướng phát triển tương lai](#-hướng-phát-triển-tương-lai)

---

##  Giới thiệu dự án

Việc giao tiếp giữa người khiếm thính và người bình thường gặp nhiều rào cản do sự khác biệt về ngôn ngữ. Dự án này được xây dựng với mục tiêu tạo ra một công cụ hỗ trợ dịch thuật ngôn ngữ ký hiệu tự động và nhanh chóng. 

Thay vì sử dụng các mô hình CNN nặng nề để phân tích từng khung hình (frame) gốc, dự án tiếp cận theo hướng trích xuất **Hand Landmarks** (khung xương bàn tay) bằng Google MediaPipe, sau đó đưa chuỗi tọa độ này vào mạng nơ-ron hồi quy **LSTM** để nắm bắt sự thay đổi của chuyển động theo thời gian (temporal features).

##  Tính năng nổi bật

* **Trích xuất đặc trưng nhẹ & nhanh:** Sử dụng MediaPipe để lấy 126 tọa độ (x, y, z) của 2 bàn tay, giảm thiểu tối đa kích thước dữ liệu đầu vào so với xử lý ảnh thô.
* **Chuẩn hóa dữ liệu thông minh:** Các điểm ảnh được chuẩn hóa theo tỷ lệ khoảng cách của bàn tay, giúp mô hình nhận diện chính xác dù người dùng đưa tay gần hay xa camera.
* **Xử lý chuỗi thời gian (Time-series):** Mô hình LSTM được thiết kế để phân tích một chuỗi gồm 120 frames liên tục, đảm bảo nhận diện chính xác các từ vựng có hành động phức tạp.
* **Inference thời gian thực (Real-time):** Tối ưu hóa pipeline để chạy mượt mà trên Webcam thông thường mà không cần GPU cấu hình cao.

##  Kiến trúc hệ thống

Hệ thống hoạt động qua 3 giai đoạn chính:
1.  **Data Extraction (Tiền xử lý):** Đọc video từ bộ WLASL -> MediaPipe trích xuất tọa độ -> Chuẩn hóa tọa độ -> Đệm (Padding) về độ dài cố định 120 frames -> Lưu thành mảng Numpy.
2.  **Model Training (Huấn luyện):** Load mảng Numpy -> Đưa vào mạng LSTM (gồm 2 lớp LSTM 128 units, kết hợp Dropout chống Overfitting) -> Phân loại đa lớp (Softmax).
3.  **Real-time Inference (Dự đoán):** Webcam đọc frame -> MediaPipe lấy tọa độ -> Đẩy vào mảng bộ nhớ đệm (Sliding Window) -> Nếu đủ 120 frames sẽ đưa vào mô hình dự đoán -> Hiển thị kết quả (từ vựng và độ tin cậy) lên màn hình bằng OpenCV.

##  Bộ dữ liệu (Dataset)

Dự án sử dụng bộ dữ liệu **WLASL**, cụ thể:
* `nslt_100.json`: Tệp metadata chứa thông tin về tập train/test và nhãn (label) của các video.
* `wlasl_class_list.txt`: Từ điển ánh xạ (mapping) giữa Class ID (số) và từ vựng tiếng Anh thực tế (ví dụ: `0 -> book`, `12 -> help`). Hệ thống tự động đọc file này để hiển thị Vietsub/Engsub lên màn hình thay vì hiện những con số khô khan.

##  Cấu trúc thư mục

```text
WLASL-Sign-Language-AI/
│
├── dataset/                     # Thư mục chứa dữ liệu
│   ├── videos/                  # Chứa các file video .mp4 của WLASL
│   ├── keypoints/               # Nơi lưu trữ file tọa độ .npy sau khi trích xuất
│   ├── nslt_100.json            # Metadata của bộ dữ liệu
│   └── wlasl_class_list.txt     # Từ điển mapping ID sang Text
│
├── config.py                    # Lưu trữ các tham số, đường dẫn và hằng số
├── utils.py                     # Chứa hàm trích xuất MediaPipe và xử lý mảng
├── 1_extract_data.py            # Chạy để chuyển video thành mảng numpy
├── 2_train_model.py             # Xây dựng và huấn luyện mạng Deep Learning LSTM
├── 3_realtime_inference.py      # Mở Webcam và chạy AI trực tiếp
│
├── requirements.txt             # Danh sách thư viện cần thiết
└── README.md                    # File tài liệu bạn đang đọc
```

##  Hướng dẫn cài đặt

**Bước 1: Clone kho lưu trữ này về máy**
```bash
git clone [https://github.com/your-username/WLASL-Sign-Language-AI.git](https://github.com/your-username/WLASL-Sign-Language-AI.git)
cd WLASL-Sign-Language-AI
```

**Bước 2: Cài đặt các thư viện phụ thuộc**
Khuyến nghị sử dụng môi trường ảo (Virtual Environment) với Python 3.8 - 3.10.
```bash
pip install -r requirements.txt
```

**Bước 3: Chuẩn bị dữ liệu**
* Tạo thư mục `dataset/videos/` và đặt các video WLASL vào đó.
* Đảm bảo file `nslt_100.json` và `wlasl_class_list.txt` đã nằm trong thư mục `dataset/`.

##  Cách sử dụng

Chạy các file theo thứ tự sau để xây dựng và kiểm thử hệ thống:

**1. Trích xuất dữ liệu từ video:**
Quá trình này sẽ lấy tọa độ tay và lưu thành `X_data.npy` và `y_data.npy`.
```bash
python 1_extract_data.py
```

**2. Huấn luyện AI:**
Quá trình này sẽ nạp dữ liệu, train model LSTM và lưu trọng số vào file `sign_language_model.h5`.
```bash
python 2_train_model.py
```

**3. Chạy Webcam nhận diện thực tế:**
Hãy đứng trước camera, thực hiện hành động và xem AI phản hồi! Bấm phím `q` để thoát ứng dụng.
```bash
python 3_realtime_inference.py
```

##  Tùy chỉnh

Bạn có thể dễ dàng thay đổi hành vi của mô hình bằng cách sửa file `config.py`:
* `NUM_CLASSES`: Đổi số lượng từ vựng AI cần học (Mặc định là 10, bạn có thể tăng lên 50, 100 tùy vào sức mạnh máy tính).
* `MAX_FRAMES`: Độ dài của chuỗi sequence (Mặc định: 120 frames).
Trong file `3_realtime_inference.py`, bạn có thể chỉnh biến `confidence > 0.5` để tăng/giảm độ khó (ngưỡng tin cậy) khi AI đưa ra quyết định.

##  Hướng phát triển tương lai

* [ ] Tích hợp MediaPipe Holistic để lấy thêm tọa độ khuôn mặt (Facial Expressions) và cơ thể (Pose), vì biểu cảm khuôn mặt rất quan trọng trong ngôn ngữ ký hiệu.
* [ ] Thay thế kiến trúc LSTM bằng Transformers để bắt ngữ cảnh thời gian dài hơn.
* [ ] Đóng gói ứng dụng thành file thực thi `.exe` hoặc giao diện Web với Streamlit/Flask.

---
*Dự án được phát triển nhằm mục đích học tập và nghiên cứu các ứng dụng của AI trong đời sống.*
```
