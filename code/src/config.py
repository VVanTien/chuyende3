import os

# ===========================================================================
# CẤU HÌNH ĐƯỜNG DẪN DỮ LIỆU
# ===========================================================================

# Dữ liệu thô gốc (3 bộ riêng lẻ)
DATA_RAW_DIR     = os.path.join("data", "raw")
DATA_RAW_DS1     = os.path.join(DATA_RAW_DIR, "dataset_1", "Travel details dataset.csv")
DATA_RAW_DS2_DIR = os.path.join(DATA_RAW_DIR, "dataset_2")
DATA_RAW_DS3_DIR = os.path.join(DATA_RAW_DIR, "dataset_3")

# Dữ liệu đã xử lý (output của pipeline)
DATA_PROCESSED_DIR  = os.path.join("data", "processed")
DATA_MERGED         = os.path.join(DATA_PROCESSED_DIR, "merged_travel_data.csv")   # Sau bước merge
DATA_PROCESSED      = os.path.join(DATA_PROCESSED_DIR, "cleaned_data.csv")         # Sau bước clean+feature

# Báo cáo phân tích dữ liệu
MISSING_VALUE_REPORT_PATH = os.path.join(DATA_PROCESSED_DIR, "missing_value_report.csv")
CORRELATION_REPORT_PATH   = os.path.join(DATA_PROCESSED_DIR, "top_correlations.csv")
DESTINATION_STATS_PATH    = os.path.join(DATA_PROCESSED_DIR, "destination_cost_stats.csv")

# ===========================================================================
# CẤU HÌNH MÔ HÌNH VÀ BIỂU ĐỒ
# ===========================================================================
MODEL_PATH_DIR      = os.path.join("outputs", "models")
MODEL_PATH          = os.path.join(MODEL_PATH_DIR, "travel_model.pkl")
MODEL_METRICS_PATH  = os.path.join(MODEL_PATH_DIR, "model_metrics.csv")
FIGURES_PATH        = os.path.join("outputs", "figures")

# Tự động tạo các thư mục đầu ra nếu chưa tồn tại
os.makedirs(DATA_PROCESSED_DIR, exist_ok=True)
os.makedirs(MODEL_PATH_DIR, exist_ok=True)
os.makedirs(FIGURES_PATH, exist_ok=True)

