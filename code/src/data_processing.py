import pandas as pd
import numpy as np
from src.config import MISSING_VALUE_REPORT_PATH, CORRELATION_REPORT_PATH, DESTINATION_STATS_PATH

def load_data(data_path) -> pd.DataFrame | None:
    """Đọc dữ liệu gốc"""
    try:
        df = pd.read_csv(data_path)
        print(f"Đã tải dữ liệu thành công: {df.shape[0]} dòng.")
        return df
    except Exception as e:
        print(f"Lỗi khi tải dữ liệu: {e}")
        return None

def missing_value_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Tạo báo cáo thống kê Missing Value (dữ liệu bị thiếu) của từng cột
    TRƯỚC KHI tiến hành làm sạch, nhằm ghi lại tình trạng ban đầu của dataset.

    Báo cáo bao gồm:
        - column         : Tên cột
        - dtype          : Kiểu dữ liệu của cột
        - total_count    : Tổng số giá trị trong cột
        - missing_count  : Số lượng giá trị bị thiếu (NaN/None)
        - missing_percent: Tỷ lệ phần trăm giá trị bị thiếu
        - non_missing    : Số lượng giá trị hợp lệ (không thiếu)

    Kết quả được lưu ra file CSV tại đường dẫn cấu hình sẵn.

    Tham số:
        df (pd.DataFrame): DataFrame gốc chưa qua làm sạch.

    Trả về:
        pd.DataFrame: Bảng tổng hợp Missing Value.
    """
    total_rows = len(df)

    report = pd.DataFrame({
        'column'         : df.columns,
        'dtype'          : df.dtypes.values,
        'total_count'    : total_rows,
        'missing_count'  : df.isnull().sum().values,
        'non_missing'    : df.notnull().sum().values,
    })

    # Tính phần trăm missing, làm tròn 2 chữ số thập phân
    report['missing_percent (%)'] = (
        (report['missing_count'] / total_rows) * 100
    ).round(2)

    # Sắp xếp giảm dần theo số lượng giá trị thiếu để dễ review
    report = report.sort_values('missing_count', ascending=False).reset_index(drop=True)

    # Lưu ra file CSV
    report.to_csv(MISSING_VALUE_REPORT_PATH, index=False)

    # In tóm tắt ra terminal
    total_missing = report['missing_count'].sum()
    cols_with_missing = (report['missing_count'] > 0).sum()
    print("-" * 50)
    print(f"  THONG KE MISSING VALUE (truoc khi clean)")
    print("-" * 50)
    print(f"  Tong so dong       : {total_rows}")
    print(f"  Tong so cot        : {len(df.columns)}")
    print(f"  Cot co missing     : {cols_with_missing}/{len(df.columns)}")
    print(f"  Tong o bi thieu    : {total_missing}")
    print(f"  Bao cao da luu tai: {MISSING_VALUE_REPORT_PATH}")
    print("-" * 50)

    return report

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Làm sạch dữ liệu"""
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

def add_features(df: pd.DataFrame) -> pd.DataFrame:
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


def save_relationship_analysis(df: pd.DataFrame) -> tuple[pd.Series, pd.DataFrame]:
    """
    Phân tích và lưu 2 báo cáo quan hệ của dữ liệu sau khi đã làm sạch:

    1. Tương quan số học với 'Total Cost':
       - Tính hệ số tương quan Pearson giữa tất cả các cột số và cột mục tiêu.
       - Sắp xếp theo giá trị tuyệt đối giảm dần (cường độ tương quan mạnh nhất lên trước).
       - Lưu top 10 vào file 'top_correlations.csv'.

    2. Thống kê chi phí theo điểm đến (Destination):
       - Tính số lượng chuyến, chi phí trung bình, trung vị, nhỏ nhất, lớn nhất cho mỗi điểm đến.
       - Sắp xếp giảm dần theo chi phí trung bình.
       - Lưu toàn bộ vào file 'destination_cost_stats.csv'.

    Tham số:
        df (pd.DataFrame): DataFrame đã qua làm sạch và feature engineering.

    Trả về:
        tuple[
            pd.Series    -> top 10 tương quan mạnh nhất với Total Cost,
            pd.DataFrame -> bảng thống kê chi phí theo từng điểm đến
        ]
    """
    target = 'Total Cost'

    # ----------------------------------------------------------------
    # PHẦN 1: TƯƠNG QUAN VỚI CHI PHÍ TỔNG (Total Cost)
    # ----------------------------------------------------------------
    numeric_corr = None
    if target in df.columns:
        # Lấy mả trận tương quan chỉ cho các cột số, lấy cột mục tiêu
        numeric_corr = (
            df.corr(numeric_only=True)[target]
            .drop(target, errors='ignore')       # Bỏ chính nó ra khỏi danh sách
            .sort_values(
                key=lambda s: s.abs(),           # Sắp xếp theo giá trị tuyệt đối
                ascending=False
            )
        )
        top_correlations = numeric_corr.head(10)

        # Lưu ra CSV (giữ lại tiêu đề cột)
        top_correlations.rename('correlation_with_Total_Cost').to_csv(
            CORRELATION_REPORT_PATH, header=True
        )
        print(f"  [1] Top tương quan đã lưu: {CORRELATION_REPORT_PATH}")
    else:
        top_correlations = pd.Series(dtype=float)
        print(f"  [!] Không tìm thấy cột '{target}', bỏ qua phân tích tương quan.")

    # ----------------------------------------------------------------
    # PHẦN 2: THỐNG KÊ CHI PHÍ THEO ĐIỂM ĐẼN (Destination)
    # ----------------------------------------------------------------
    destination_stats = pd.DataFrame()  # mặc định nếu không có cột
    if 'Destination' in df.columns and target in df.columns:
        destination_stats = (
            df.groupby('Destination', dropna=False)[target]
            .agg(
                count='count',      # Số chuyến đến điểm này
                mean='mean',        # Chi phí trung bình
                median='median',    # Chi phí trung vị
                min='min',          # Chi phí thấp nhất
                max='max'           # Chi phí cao nhất
            )
            .round(2)
            .sort_values('mean', ascending=False)   # Điểm đến đắt tiền nhất lên trước
            .reset_index()
        )
        destination_stats.to_csv(DESTINATION_STATS_PATH, index=False)
        print(f"  [2] Thống kê theo điểm đến đã lưu: {DESTINATION_STATS_PATH}")
    else:
        print("  [!] Không tìm thấy cột 'Destination' hoặc 'Total Cost', bỏ qua.")

    return top_correlations, destination_stats