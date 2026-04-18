import sys
import os
# Thêm đường dẫn thư mục gốc vào sys.path để tránh lỗi ModuleNotFoundError: No module named 'src'
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_processing import load_data, clean_data, add_features, missing_value_summary, save_relationship_analysis
from src.visualization import visualization
from src.model import train_model
from src.config import DATA_PROCESSED, DATA_RAW

def run_pipeline():
    print("--- Bắt đầu quy trình Phân tích Dữ liệu Du lịch ---")
    
    # Bước 1: Thu thập dữ liệu
    df = load_data(DATA_RAW)

    
    if df is not None:
        # Bước 1b: Thống kê Missing Value TRƯỚC khi làm sạch và lưu báo cáo CSV
        missing_value_summary(df)

        # Bước 2: Làm sạch
        print("Đang làm sạch dữ liệu...")
        cleaned_df = clean_data(df)
        
        # Bước 3: Trích xuất đặc trưng mới
        print("Đang tạo thêm các đặc trưng phân tích...")
        processed_df = add_features(cleaned_df)
        
        # Bước 3b: Phân tích quan hệ: Tương quan và thống kê theo điểm đến
        print("Đang phân tích các mối quan hệ dữ liệu...")
        save_relationship_analysis(processed_df)
        
        # Lưu file dữ liệu đã xử lý
        processed_df.to_csv(DATA_PROCESSED, index=False)
        print(f"Đã lưu tập dữ liệu làm sạch tại: {DATA_PROCESSED}")
        
        # Bước 4: Visualization (Vẽ biểu đồ phân tích insight)
        visualization(processed_df)
        
        # Bước 5: Huấn luyện mô hình và lưu metrics
        print("\nĐang huấn luyện mô hình dự đoán...")
        train_model(processed_df)
        
    print("--- Hoàn thành pipeline ---")

if __name__ == "__main__":
    run_pipeline()

