import pandas as pd
import numpy as np

def clean_travel_dataset(df):
    # Loại bỏ dòng trống hoàn toàn
    df = df.dropna(how='all')
    
    # Chuẩn hóa tên cột để dễ làm việc (loại bỏ khoảng trắng thừa)
    df.columns = df.columns.str.strip()
    
    # Hàm con để làm sạch cột chi phí (loại bỏ $, phẩy, ký tự chữ và NaN)
    def clean_cost(val):
        if pd.isna(val):
            return np.nan
        if isinstance(val, (int, float)):
            return float(val)
        val = str(val).replace('$', '').replace(',', '').replace('USD', '').strip()
        try:
            return float(val)
        except ValueError:
            return np.nan

    # Áp dụng hàm làm sạch cho các cột cost
    cost_cols = ['Accommodation cost', 'Transportation cost']
    for col in cost_cols:
        if col in df.columns:
            df[col] = df[col].apply(clean_cost)
            
    # Chuyển đổi ngày tháng
    date_cols = ['Start date', 'End date']
    for col in date_cols:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors='coerce')
    
    # Xử lý missing value
    # Đối với cột dạng số, ta có thể dùng giá trị trung vị (median)
    for col in df.select_dtypes(include=[np.number]).columns:
        df[col] = df[col].fillna(df[col].median())
        
    # Đối với dạng chuỗi, điền "Unknown" hoặc mode
    for col in df.select_dtypes(include=['object']).columns:
        df[col] = df[col].fillna(df[col].mode()[0] if not df[col].mode().empty else 'Unknown')
        
    return df