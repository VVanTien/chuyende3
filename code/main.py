import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_processing import (
    merge_all_datasets,
    load_data,
    missing_value_summary,
    clean_data,
    add_features,
    save_relationship_analysis,
)
from src.visualization import visualization, plot_missing_values
from src.model import train_model
from src.config import DATA_MERGED, DATA_PROCESSED
from src.spinner import Spinner


def run_pipeline():
    print("\n" + "=" * 55)
    print("  PIPELINE: PHAN TICH DU LIEU DU LICH")
    print("=" * 55)

    # ------------------------------------------------------------------
    # BƯỚC 0: Merge 3 dataset thô → data/processed/merged_travel_data.csv
    # ------------------------------------------------------------------
    print("\n[Buoc 0] Merge datasets...")
    with Spinner("Dang gop 3 datasets thanh 1 schema chung", style='bounce'):
        merged_df = merge_all_datasets()

    # ------------------------------------------------------------------
    # BƯỚC 1: Load dữ liệu đã merge
    # ------------------------------------------------------------------
    print("\n[Buoc 1] Tai du lieu da merge...")
    df = load_data(DATA_MERGED)
    if df is None:
        print("  [LOI] Khong the tai du lieu. Ket thuc pipeline.")
        return

    # ------------------------------------------------------------------
    # BƯỚC 2: Báo cáo Missing Value & vẽ biểu đồ (trước khi làm sạch)
    # ------------------------------------------------------------------
    print("\n[Buoc 2] Phan tich Missing Value...")
    missing_report = missing_value_summary(df)
    with Spinner("Dang ve bieu do Missing Value", style='bar'):
        plot_missing_values(missing_report)

    # ------------------------------------------------------------------
    # BƯỚC 3: Làm sạch dữ liệu
    # ------------------------------------------------------------------
    print("\n[Buoc 3] Lam sach du lieu...")
    with Spinner("Dang xu ly va chuan hoa du lieu", style='dots'):
        cleaned_df = clean_data(df)

    # ------------------------------------------------------------------
    # BƯỚC 4: Feature Engineering & Phân tích quan hệ
    # ------------------------------------------------------------------
    print("\n[Buoc 4] Feature Engineering...")
    with Spinner("Dang tao them cac dac trung moi", style='dots'):
        processed_df = add_features(cleaned_df)

    print("\n[Buoc 4b] Phan tich tuong quan & thong ke diem den...")
    with Spinner("Dang tinh toan tuong quan Pearson", style='arrow'):
        save_relationship_analysis(processed_df)

    # Lưu file đã xử lý hoàn chỉnh
    with Spinner(f"Dang luu du lieu sach vao {DATA_PROCESSED}", style='classic'):
        processed_df.to_csv(DATA_PROCESSED, index=False)

    # ------------------------------------------------------------------
    # BƯỚC 5: Visualization — vẽ toàn bộ biểu đồ EDA
    # ------------------------------------------------------------------
    print("\n[Buoc 5] Ve toan bo bieu do EDA (co the mat vai phut)...")
    with Spinner("Dang render va luu cac bieu do PNG (300 DPI)", style='bounce'):
        visualization(processed_df)

    # ------------------------------------------------------------------
    # BƯỚC 6: Huấn luyện mô hình học máy
    # ------------------------------------------------------------------
    # Bước 6 tự có logging chi tiết từ model.py nên không bọc spinner toàn bộ
    print("\n[Buoc 6] Huan luyen va so sanh cac mo hinh hoc may...")
    train_model(processed_df)

    print("\n" + "=" * 55)
    print("  ✔  HOAN THANH PIPELINE!")
    print("=" * 55)


if __name__ == "__main__":
    run_pipeline()