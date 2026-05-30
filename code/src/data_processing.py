# =============================================================================
# Module: data_processing.py
# Chức năng: Xử lý toàn bộ dữ liệu đầu vào — gồm 3 giai đoạn chính:
#   1. MERGE    : Gộp 3 bộ dataset thô về chung 1 schema chuẩn
#   2. CLEAN    : Làm sạch, chuẩn hóa và điền missing value
#   3. FEATURES : Tạo thêm các đặc trưng (Feature Engineering) phục vụ EDA & ML
# =============================================================================

import os
import pandas as pd
import numpy as np

from src.config import (
    DATA_RAW_DS1, DATA_RAW_DS2_DIR, DATA_RAW_DS3_DIR, DATA_MERGED,
    MISSING_VALUE_REPORT_PATH, CORRELATION_REPORT_PATH, DESTINATION_STATS_PATH
)


# =============================================================================
# PHẦN 1: MERGE — Gộp 3 dataset về schema chung
# =============================================================================

def _load_ds1() -> pd.DataFrame:
    """
    Dataset 1 — Travel details dataset (139 dòng).
    Columns: Trip ID, Destination, Start date, End date, Duration (days),
             Traveler name/age/gender/nationality, Accommodation type/cost,
             Transportation type/cost.
    """
    df = pd.read_csv(DATA_RAW_DS1)

    def _clean_cost(val):
        if pd.isna(val): return np.nan
        if isinstance(val, (int, float)): return float(val)
        return float(str(val).replace('$', '').replace(',', '').replace('USD', '').strip())

    df['Accommodation cost']  = df['Accommodation cost'].apply(_clean_cost)
    df['Transportation cost'] = df['Transportation cost'].apply(_clean_cost)
    df['Start date'] = pd.to_datetime(df['Start date'], errors='coerce')

    result = pd.DataFrame({
        'Destination'          : df['Destination'],
        'Start date'           : df['Start date'],
        'Duration (days)'      : pd.to_numeric(df['Duration (days)'], errors='coerce'),
        'Traveler age'         : pd.to_numeric(df['Traveler age'], errors='coerce'),
        'Traveler gender'      : df['Traveler gender'].str.strip().str.capitalize(),
        'Traveler nationality' : df['Traveler nationality'].str.strip() if 'Traveler nationality' in df.columns else np.nan,
        'Accommodation type'   : df['Accommodation type'].str.strip(),
        'Accommodation cost'   : df['Accommodation cost'],
        'Transportation type'  : df['Transportation type'].str.strip(),
        'Transportation cost'  : df['Transportation cost'],
        'source'               : 'dataset_1'
    })
    print(f"  [Dataset 1] Loaded: {len(result)} rows")
    return result


def _load_ds2() -> pd.DataFrame:
    """
    Dataset 2 — ArgoDatathon 2019 (flights + hotels + users, ~28k dòng sau JOIN).
    Cần JOIN 3 bảng: flights x hotels (theo travelCode+userCode) x users (theo userCode).
    """
    flights = pd.read_csv(os.path.join(DATA_RAW_DS2_DIR, "flights.csv"))
    hotels  = pd.read_csv(os.path.join(DATA_RAW_DS2_DIR, "hotels.csv"))
    users   = pd.read_csv(os.path.join(DATA_RAW_DS2_DIR, "users.csv"))

    # JOIN flights + hotels (inner) → chỉ giữ chuyến có đủ cả 2 bảng
    travel = pd.merge(
        flights, hotels,
        on=['travelCode', 'userCode'],
        how='inner',
        suffixes=('_flight', '_hotel')
    )
    # JOIN với users (left) → bổ sung thông tin hành khách
    travel = pd.merge(travel, users, left_on='userCode', right_on='code', how='left')

    travel['date_flight'] = pd.to_datetime(travel['date_flight'], errors='coerce', dayfirst=False)

    # Tổng chi phí vé máy bay theo travelCode (gộp nhiều chặng)
    flight_cost = flights.groupby(['travelCode', 'userCode'])['price'].sum().reset_index()
    flight_cost.rename(columns={'price': 'total_flight_cost'}, inplace=True)
    travel = pd.merge(travel, flight_cost, on=['travelCode', 'userCode'], how='left')

    # Mapping flightType → nhãn thân thiện
    flight_type_map = {
        'firstClass': 'First Class Flight',
        'economic'  : 'Economy Flight',
        'business'  : 'Business Flight',
    }
    travel['flightType_mapped'] = (
        travel['flightType'].str.strip()
        .map(flight_type_map)
        .fillna(travel['flightType'])
    )

    # Sau 3-way merge: name_x = hotel name, name_y = user name
    result = pd.DataFrame({
        'Destination'          : travel['to'].str.strip(),
        'Start date'           : travel['date_flight'],
        'Duration (days)'      : pd.to_numeric(travel['days'], errors='coerce'),
        'Traveler age'         : pd.to_numeric(travel['age'], errors='coerce'),
        'Traveler gender'      : travel['gender'].str.strip().str.capitalize(),
        'Traveler nationality' : np.nan,
        'Accommodation type'   : travel['name_x'].str.strip(),
        'Accommodation cost'   : pd.to_numeric(travel['total'], errors='coerce'),
        'Transportation type'  : travel['flightType_mapped'],
        'Transportation cost'  : pd.to_numeric(travel['total_flight_cost'], errors='coerce'),
        'source'               : 'dataset_2'
    })

    # Loại bỏ duplicate phát sinh từ nhiều chặng bay cùng 1 chuyến
    result = result.drop_duplicates(subset=['Destination', 'Start date', 'Accommodation cost'])
    print(f"  [Dataset 2] Loaded: {len(result)} rows (after dedup)")
    return result


def _load_ds3() -> pd.DataFrame:
    """
    Dataset 3 — Greece Travel Data (3000 dòng).
    Columns: Trip #no., Duration, Cost of Travel(Entire Trip), Mode of Travel,
             Stay, Age, Sex, Nationality, Date of Travel.
    Chi phí tổng được chia ước tính: 60% lưu trú / 40% di chuyển.
    """
    path = os.path.join(DATA_RAW_DS3_DIR, "travel_tourism_dataset.csv")
    df = pd.read_csv(path)

    gender_map = {
        'Female': 'Female', 'female': 'Female',
        'Male'  : 'Male',   'male'  : 'Male',
        'Non-Binary': 'Non-binary',
    }
    transport_map = {
        'Flight': 'Flight', 'Car': 'Car',
        'Roadtrip': 'Road Trip', 'Train': 'Train', 'Bus': 'Bus',
    }
    stay_map = {
        'Hotel': 'Hotel', 'Airbnb': 'Airbnb',
        'Hostel': 'Hostel', 'Resort': 'Resort',
    }

    total_cost  = pd.to_numeric(df['Cost of Travel(Entire Trip)'], errors='coerce')
    accom_cost  = (total_cost * 0.60).round(2)
    transp_cost = (total_cost * 0.40).round(2)

    result = pd.DataFrame({
        'Destination'          : 'Greece',
        'Start date'           : pd.to_datetime(df['Date of Travel'], errors='coerce'),
        'Duration (days)'      : pd.to_numeric(df['Duration'], errors='coerce'),
        'Traveler age'         : pd.to_numeric(df['Age'], errors='coerce'),
        'Traveler gender'      : df['Sex'].str.strip().map(gender_map).fillna(df['Sex'].str.strip()),
        'Traveler nationality' : df['Nationality'].str.strip() if 'Nationality' in df.columns else np.nan,
        'Accommodation type'   : df['Stay'].str.strip().map(stay_map).fillna(df['Stay']),
        'Accommodation cost'   : accom_cost,
        'Transportation type'  : df['Mode of Travel'].str.strip().map(transport_map).fillna(df['Mode of Travel']),
        'Transportation cost'  : transp_cost,
        'source'               : 'dataset_3'
    })
    print(f"  [Dataset 3] Loaded: {len(result)} rows")
    return result


def merge_all_datasets() -> pd.DataFrame:
    """
    Gộp (concat theo chiều dọc) cả 3 tập dữ liệu thô về chung 1 schema chuẩn.
    Kết quả được lưu vào data/processed/merged_travel_data.csv.

    Trả về:
        pd.DataFrame: DataFrame đã merge, chưa làm sạch.
    """
    print("=" * 55)
    print("  BUOC 0: MERGE 3 DATASETS")
    print("=" * 55)

    df1 = _load_ds1()
    df2 = _load_ds2()
    df3 = _load_ds3()

    merged = pd.concat([df1, df2, df3], ignore_index=True)

    # Loại bỏ dòng trùng hoàn toàn
    before = len(merged)
    merged = merged.drop_duplicates()
    removed = before - len(merged)

    print("-" * 55)
    print(f"  Tong so dong sau merge : {len(merged)}")
    print(f"  Phan bo theo nguon:")
    for src, cnt in merged['source'].value_counts().items():
        print(f"    {src:<12}: {cnt:>6} dong")
    if removed:
        print(f"  Da loai bo {removed} dong trung lap.")

    merged.to_csv(DATA_MERGED, index=False)
    print(f"  File merged da luu tai: {DATA_MERGED}")
    print("=" * 55)
    return merged


# =============================================================================
# PHẦN 2: LOAD — Đọc dữ liệu (sau khi đã merge)
# =============================================================================

def load_data(data_path: str) -> pd.DataFrame | None:
    """Đọc file CSV dữ liệu (dùng cho cả file gốc lẫn file đã merge)."""
    try:
        df = pd.read_csv(data_path)
        # Bỏ cột Trip ID nếu có (cột ID không mang thông tin phân tích)
        if "Trip ID" in df.columns:
            df = df.drop(columns=["Trip ID"])
        print(f"  Da tai du lieu: {df.shape[0]} dong x {df.shape[1]} cot  ({data_path})")
        return df
    except Exception as e:
        print(f"  [LOI] Khong the tai du lieu: {e}")
        return None


# =============================================================================
# PHẦN 3: MISSING VALUE — Báo cáo trước khi làm sạch
# =============================================================================

def missing_value_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Tạo báo cáo thống kê Missing Value của từng cột TRƯỚC KHI làm sạch.
    Kết quả lưu vào MISSING_VALUE_REPORT_PATH.

    Trả về:
        pd.DataFrame: Bảng tổng hợp missing value.
    """
    total_rows = len(df)
    report = pd.DataFrame({
        'column'         : df.columns,
        'dtype'          : df.dtypes.values,
        'total_count'    : total_rows,
        'missing_count'  : df.isnull().sum().values,
        'non_missing'    : df.notnull().sum().values,
    })
    report['missing_percent (%)'] = (report['missing_count'] / total_rows * 100).round(2)
    report = report.sort_values('missing_count', ascending=False).reset_index(drop=True)
    report.to_csv(MISSING_VALUE_REPORT_PATH, index=False)

    total_missing     = report['missing_count'].sum()
    cols_with_missing = (report['missing_count'] > 0).sum()
    print("-" * 50)
    print("  THONG KE MISSING VALUE (truoc khi clean)")
    print("-" * 50)
    print(f"  Tong so dong       : {total_rows}")
    print(f"  Tong so cot        : {len(df.columns)}")
    print(f"  Cot co missing     : {cols_with_missing}/{len(df.columns)}")
    print(f"  Tong o bi thieu    : {total_missing}")
    print(f"  Bao cao da luu tai: {MISSING_VALUE_REPORT_PATH}")
    print("-" * 50)
    return report


# =============================================================================
# PHẦN 4: CLEAN — Làm sạch dữ liệu
# =============================================================================

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Làm sạch DataFrame:
      - Loại bỏ dòng trống hoàn toàn.
      - Chuẩn hóa tên cột (strip whitespace).
      - Làm sạch cột chi phí (loại bỏ $, dấu phẩy).
      - Chuyển đổi cột ngày sang datetime.
      - Điền missing: cột số → Median, cột chuỗi → Mode.
    """
    df = df.dropna(how='all')
    df.columns = df.columns.str.strip()

    def _clean_cost(val):
        if pd.isna(val): return np.nan
        if isinstance(val, (int, float)): return float(val)
        val = str(val).replace('$', '').replace(',', '').replace('USD', '').strip()
        try:
            return float(val)
        except ValueError:
            return np.nan

    for col in ['Accommodation cost', 'Transportation cost']:
        if col in df.columns:
            df[col] = df[col].apply(_clean_cost)

    for col in ['Start date', 'End date']:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors='coerce')

    # Điền missing value — cột số dùng median, cột chuỗi dùng mode
    for col in df.select_dtypes(include=[np.number]).columns:
        df[col] = df[col].fillna(df[col].median())

    for col in df.select_dtypes(include=['object']).columns:
        mode_val = df[col].mode()
        df[col] = df[col].fillna(mode_val[0] if not mode_val.empty else 'Unknown')

    return df


# =============================================================================
# PHẦN 5: FEATURE ENGINEERING — Tạo đặc trưng mới
# =============================================================================

def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Tạo thêm các đặc trưng mới phục vụ EDA và huấn luyện mô hình:
      1. Total Cost       = Accommodation cost + Transportation cost
      2. Cost per day     = Total Cost / Duration
      3. Accommodation cost per day
      4. Travel Month / Year / Quarter / Season
      5. Age Group        (phân nhóm tuổi)
      6. Duration Group   (phân nhóm độ dài chuyến)
      7. Destination City / Country (tách từ cột Destination)
    """
    # 1. Tổng chi phí
    cost_cols = ['Accommodation cost', 'Transportation cost']
    if all(c in df.columns for c in cost_cols):
        df['Total Cost'] = df['Accommodation cost'] + df['Transportation cost']

    # 2. Chi phí mỗi ngày
    if 'Total Cost' in df.columns and 'Duration (days)' in df.columns:
        dur_safe = df['Duration (days)'].replace(0, np.nan)
        df['Cost per day']                = (df['Total Cost'] / dur_safe).round(2)
        df['Accommodation cost per day']  = (df['Accommodation cost'] / dur_safe).round(2)

    # 3. Tách thông tin thời gian
    if 'Start date' in df.columns and pd.api.types.is_datetime64_any_dtype(df['Start date']):
        df['Travel Month']   = df['Start date'].dt.month
        df['Travel Year']    = df['Start date'].dt.year
        df['Travel Quarter'] = df['Start date'].dt.quarter

        season_map = {
            12: 'Mùa Đông', 1: 'Mùa Đông', 2: 'Mùa Đông',
            3: 'Mùa Xuân',  4: 'Mùa Xuân', 5: 'Mùa Xuân',
            6: 'Mùa Hè',    7: 'Mùa Hè',   8: 'Mùa Hè',
            9: 'Mùa Thu',   10: 'Mùa Thu',  11: 'Mùa Thu',
        }
        df['Travel Season'] = df['Travel Month'].map(season_map)

    # 4. Phân nhóm tuổi
    if 'Traveler age' in df.columns:
        bins   = [0, 18, 25, 35, 50, 65, 100]
        labels = ['Trẻ em (<18)', 'Thanh niên (18-25)', 'Người trưởng thành (26-35)',
                  'Trung niên (36-50)', 'Trung cao niên (51-65)', 'Cao niên (>65)']
        df['Age Group'] = pd.cut(df['Traveler age'], bins=bins, labels=labels, right=True)

    # 5. Phân nhóm độ dài chuyến đi
    if 'Duration (days)' in df.columns:
        dur_bins   = [0, 3, 7, 14, 9999]
        dur_labels = ['Ngắn ngày (<=3)', 'Vừa (4-7)', 'Dài ngày (8-14)', 'Rất dài (>14)']
        df['Duration Group'] = pd.cut(df['Duration (days)'], bins=dur_bins, labels=dur_labels, right=True)

    # 6. Tách Thành phố / Quốc gia từ Destination
    if 'Destination' in df.columns:
        split_dest = df['Destination'].str.split(',', n=1, expand=True)
        if split_dest.shape[1] == 2:
            df['Destination City']    = split_dest[0].str.strip()
            df['Destination Country'] = split_dest[1].str.strip()
            df['Destination Country'] = df['Destination Country'].fillna(df['Destination City'])
        else:
            df['Destination Country'] = df['Destination'].str.strip()
            df['Destination City']    = 'Unknown'

    return df


# =============================================================================
# PHẦN 6: PHÂN TÍCH QUAN HỆ — Lưu báo cáo tương quan & thống kê điểm đến
# =============================================================================

def save_relationship_analysis(df: pd.DataFrame) -> tuple[pd.Series, pd.DataFrame]:
    """
    Phân tích và lưu 2 báo cáo:
      1. Top 10 tương quan Pearson với cột 'Total Cost' → top_correlations.csv
      2. Thống kê chi phí theo từng Destination        → destination_cost_stats.csv

    Trả về:
        tuple[pd.Series, pd.DataFrame]
    """
    target = 'Total Cost'

    # --- Tương quan ---
    top_correlations = pd.Series(dtype=float)
    if target in df.columns:
        top_correlations = (
            df.corr(numeric_only=True)[target]
            .drop(target, errors='ignore')
            .sort_values(key=lambda s: s.abs(), ascending=False)
            .head(10)
        )
        top_correlations.rename('correlation_with_Total_Cost').to_csv(
            CORRELATION_REPORT_PATH, header=True
        )
        print(f"  [1] Top tuong quan da luu: {CORRELATION_REPORT_PATH}")
    else:
        print(f"  [!] Khong tim thay cot '{target}', bo qua phan tich tuong quan.")

    # --- Thống kê theo điểm đến ---
    destination_stats = pd.DataFrame()
    if 'Destination' in df.columns and target in df.columns:
        destination_stats = (
            df.groupby('Destination', dropna=False)[target]
            .agg(count='count', mean='mean', median='median', min='min', max='max')
            .round(2)
            .sort_values('mean', ascending=False)
            .reset_index()
        )
        destination_stats.to_csv(DESTINATION_STATS_PATH, index=False)
        print(f"  [2] Thong ke diem den da luu: {DESTINATION_STATS_PATH}")
    else:
        print("  [!] Khong tim thay cot 'Destination' hoac 'Total Cost', bo qua.")

    return top_correlations, destination_stats