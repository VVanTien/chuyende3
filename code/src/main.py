import sys
import os
# Thêm đường dẫn thư mục gốc vào sys.path để tránh lỗi ModuleNotFoundError: No module named 'src'
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.ingestion.load_csv import load_travel_data
from src.preprocessing.clean import clean_travel_dataset
from src.preprocessing.feature_engineering import add_features
from src.analysis.eda import run_eda
from src.config.config import DATA_PROCESSED

def run_pipeline():
    print("--- Bắt đầu quy trình Phân tích Dữ liệu Du lịch ---")
    
    # Bước 1: Thu thập dữ liệu
    raw_df = load_travel_data()
    
    if raw_df is not None:
        # Bước 2: Làm sạch
        print("Đang làm sạch dữ liệu...")
        cleaned_df = clean_travel_dataset(raw_df)
        
        # Bước 3: Trích xuất đặc trưng mới
        print("Đang tạo thêm các đặc trưng phân tích...")
        processed_df = add_features(cleaned_df)
        
        # Lưu file dữ liệu đã xử lý
        processed_df.to_csv(DATA_PROCESSED, index=False)
        print(f"Đã lưu tập dữ liệu làm sạch tại: {DATA_PROCESSED}")
        
        # Bước 4: Visualization (Vẽ biểu đồ phân tích insight)
        run_eda(processed_df)
        
    print("--- Hoàn thành pipeline ---")

if __name__ == "__main__":
    run_pipeline()