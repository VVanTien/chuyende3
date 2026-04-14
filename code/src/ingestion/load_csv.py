import pandas as pd
from src.config.config import DATA_RAW

def load_travel_data():
    try:
        df = pd.read_csv(DATA_RAW)
        print(f"Đã tải dữ liệu thành công: {df.shape[0]} dòng.")
        return df
    except Exception as e:
        print(f"Lỗi khi tải dữ liệu: {e}")
        return None