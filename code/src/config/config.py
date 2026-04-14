import os

# Đường dẫn gốc của dự án
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

DATA_RAW = os.path.join(BASE_DIR, "data", "raw", "Travel details dataset.csv")
DATA_PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")
DATA_PROCESSED = os.path.join(DATA_PROCESSED_DIR, "cleaned_data.csv")

MODEL_PATH_DIR = os.path.join(BASE_DIR, "outputs", "models")
MODEL_PATH = os.path.join(MODEL_PATH_DIR, "travel_model.pkl")

FIGURES_PATH = os.path.join(BASE_DIR, "outputs", "figures")

# Tự động tạo các thư mục đầu ra nếu chưa tồn tại
os.makedirs(DATA_PROCESSED_DIR, exist_ok=True)
os.makedirs(MODEL_PATH_DIR, exist_ok=True)
os.makedirs(FIGURES_PATH, exist_ok=True)