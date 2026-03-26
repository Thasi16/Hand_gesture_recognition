from pathlib import Path

# Đường dẫn thư mục
DATASET_PATH = Path(r"C:\Users\Admin\Desktop\project2\dataset")
VIDEO_PATH = DATASET_PATH / "videos"
JSON_PATH = DATASET_PATH / "nslt_100.json"
KEYPOINT_PATH = DATASET_PATH / "keypoints"
DICT_PATH = DATASET_PATH / "wlasl_class_list.txt"

# Tham số cấu hình chung
MAX_FRAMES = 120
NUM_CLASSES = 10
MODEL_PATH = "sign_language_model.h5"

# Tạo thư mục nếu chưa có
KEYPOINT_PATH.mkdir(exist_ok=True, parents=True)
