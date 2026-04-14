import pandas as pd

def add_features(df):
    """
    Tạo thêm các đặc trưng hữu dụng cho việc thống kê và vẽ biểu đồ.
    """
    # Đảm bảo có cột Total Cost
    cost_cols = ['Accommodation cost', 'Transportation cost']
    if all(col in df.columns for col in cost_cols):
        df['Total Cost'] = df['Accommodation cost'] + df['Transportation cost']
    
    # Trích xuất tháng và năm từ Start date
    if 'Start date' in df.columns and pd.api.types.is_datetime64_any_dtype(df['Start date']):
        df['Travel Month'] = df['Start date'].dt.month
        df['Travel Year'] = df['Start date'].dt.year
        df['Travel Month Name'] = df['Start date'].dt.month_name()
    
    # Tạo phân nhóm độ tuổi
    if 'Traveler age' in df.columns:
        bins = [0, 18, 25, 35, 50, 65, 100]
        labels = ['Trẻ em (<18)', 'Thanh niên (18-25)', 'Người trưởng thành (26-35)', 'Trung niên (36-50)', 'Trung cao niên (51-65)', 'Cao niên (>65)']
        df['Age Group'] = pd.cut(df['Traveler age'], bins=bins, labels=labels, right=True)
        
    return df
