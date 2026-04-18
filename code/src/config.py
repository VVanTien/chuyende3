import os

# Đường dẫn đến các file và thư mục
DATA_RAW = os.path.join("data", "raw", "Travel details dataset.csv")
DATA_PROCESSED_DIR = os.path.join("data", "processed")
DATA_PROCESSED = os.path.join(DATA_PROCESSED_DIR, "cleaned_data.csv")
MISSING_VALUE_REPORT_PATH = os.path.join(DATA_PROCESSED_DIR, "missing_value_report.csv")
CORRELATION_REPORT_PATH = os.path.join(DATA_PROCESSED_DIR, "top_correlations.csv")
DESTINATION_STATS_PATH = os.path.join(DATA_PROCESSED_DIR, "destination_cost_stats.csv")
MODEL_PATH_DIR = os.path.join("outputs", "models")
MODEL_PATH = os.path.join(MODEL_PATH_DIR, "travel_model.pkl")
MODEL_METRICS_PATH = os.path.join(MODEL_PATH_DIR, "model_metrics.csv")
FIGURES_PATH = os.path.join("outputs", "figures")

# Tự động tạo các thư mục đầu ra nếu chưa tồn tại
os.makedirs(DATA_PROCESSED_DIR, exist_ok=True)
os.makedirs(MODEL_PATH_DIR, exist_ok=True)
os.makedirs(FIGURES_PATH, exist_ok=True)

