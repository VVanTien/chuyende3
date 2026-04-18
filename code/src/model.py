# =============================================================================
# Module: model.py
# Chức năng: Huấn luyện mô hình học máy Random Forest để dự đoán chi phí lưu trú
#            của các chuyến đi du lịch. Sau khi huấn luyện, tự động đánh giá mô hình
#            và lưu kết quả ra file .pkl (mô hình) và .csv (thông số đánh giá).
# =============================================================================

import pandas as pd                                   # Xử lý dữ liệu dạng bảng
import numpy as np                                    # Tính toán số học (căn bậc 2 RMSE)
import joblib                                         # Lưu/tải mô hình ra file nhị phân .pkl

# --- Thư viện sklearn: Huấn luyện và đánh giá mô hình ---
from sklearn.model_selection import train_test_split  # Chia dữ liệu thành tập huấn luyện và kiểm tra
from sklearn.ensemble import RandomForestRegressor    # Thuật toán: Rừng Ngẫu Nhiên (hồi quy)
from sklearn.preprocessing import OneHotEncoder       # Mã hóa các cột dạng chuỗi thành dạng số nhị phân
from sklearn.compose import ColumnTransformer         # Áp dụng các bộ mã hóa khác nhau cho từng nhóm cột
from sklearn.pipeline import Pipeline                 # Gộp tiền xử lý và mô hình vào một quy trình duy nhất
from sklearn.metrics import (
    mean_absolute_error,    # MAE:  Trung bình giá trị tuyệt đối sai số dự đoán
    mean_squared_error,     # MSE:  Trung bình bình phương sai số (dùng để tính RMSE)
    r2_score                # R2:   Hệ số xác định, đo mức độ phù hợp của mô hình (1.0 là hoàn hảo)
)

from src.config import MODEL_PATH, MODEL_METRICS_PATH  # Đường dẫn lưu file mô hình và metrics


def train_model(df: pd.DataFrame) -> Pipeline:
    """
    Thực hiện toàn bộ quy trình huấn luyện mô hình:
      1. Chọn đặc trưng (Feature Selection)
      2. Tiền xử lý dữ liệu (Preprocessing)
      3. Huấn luyện mô hình RandomForest (Training)
      4. Đánh giá trên tập kiểm tra (Evaluation)
      5. Lưu mô hình và metrics ra file (Persistence)

    Tham số:
        df (pd.DataFrame): DataFrame đã được làm sạch và thêm đặc trưng.

    Trả về:
        Pipeline: Đối tượng Pipeline đã huấn luyện, sẵn sàng để dự đoán.
    """

    # =========================================================================
    # BƯỚC 1: XÁC ĐỊNH ĐẶC TRƯNG ĐẦU VÀO VÀ NHÃN MỤC TIÊU
    # =========================================================================

    # Các đặc trưng dạng SỐ (numeric): không cần mã hóa, dùng trực tiếp
    feature_num = ['Traveler age', 'Duration (days)']

    # Các đặc trưng dạng CHUỖI (categorical): cần mã hóa OneHot thành dạng số
    feature_cat = ['Destination', 'Traveler gender', 'Accommodation type', 'Transportation type']

    # Nhãn mục tiêu (target): giá trị cần dự đoán
    target = 'Accommodation cost'

    # Lọc ra chỉ các cột thực sự tồn tại trong DataFrame
    # (Đề phòng trường hợp pipeline bị chạy với DataFrame thiếu cột)
    feature_num = [c for c in feature_num if c in df.columns]
    feature_cat = [c for c in feature_cat if c in df.columns]
    all_features = feature_num + feature_cat  # Tổng hợp toàn bộ đặc trưng

    # =========================================================================
    # BƯỚC 2: CHUẨN BỊ DỮ LIỆU HUẤN LUYỆN VÀ KIỂM TRA
    # =========================================================================

    # Chỉ giữ lại các cột cần thiết và loại bỏ các hàng bị thiếu dữ liệu
    df_model = df[all_features + [target]].dropna()

    X = df_model[all_features]   # Ma trận đặc trưng (features matrix)
    y = df_model[target]         # Vector nhãn (label vector)

    # Chia dữ liệu: 80% để huấn luyện, 20% để kiểm tra
    # random_state=42 đảm bảo kết quả tái hiện được (reproducibility)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # =========================================================================
    # BƯỚC 3: XÂY DỰNG PIPELINE TIỀN XỬ LÝ + MÔ HÌNH
    # =========================================================================

    # Bộ tiền xử lý (Preprocessor):
    # - Với các cột CHUỖI (feature_cat): Áp dụng OneHotEncoder
    #     + handle_unknown='ignore': Bỏ qua giá trị lạ chưa thấy khi huấn luyện
    #     + sparse_output=False: Trả về ma trận dense thay vì sparse (dễ xử lý hơn)
    # - Với các cột SỐ (feature_num): remainder='passthrough' -> giữ nguyên, không thay đổi
    preprocessor = ColumnTransformer(transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), feature_cat)
    ], remainder='passthrough')

    # Pipeline gộp 2 bước thành 1 quy trình liền mạch:
    #   Bước 1 ('preprocessor'): Biến dữ liệu thô -> dữ liệu đã mã hóa
    #   Bước 2 ('model'): Đưa dữ liệu đã mã hóa vào RandomForest để học
    #
    # RandomForestRegressor:
    #   - n_estimators=200: Dùng 200 cây quyết định (càng nhiều cây -> càng ổn định)
    #   - random_state=42:  Seed cố định để kết quả nhất quán giữa các lần chạy
    #   - n_jobs=-1:        Tận dụng toàn bộ CPU để tăng tốc độ huấn luyện
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('model', RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1))
    ])

    # =========================================================================
    # BƯỚC 4: HUẤN LUYỆN VÀ DỰ ĐOÁN
    # =========================================================================

    # Fit pipeline trên tập huấn luyện:
    # sklearn tự động chạy preprocessor.fit_transform(X_train) -> rồi model.fit(...)
    pipeline.fit(X_train, y_train)

    # Dự đoán trên tập kiểm tra (tập mô hình chưa từng thấy)
    y_pred = pipeline.predict(X_test)

    # =========================================================================
    # BƯỚC 5: ĐÁNH GIÁ CHẤT LƯỢNG MÔ HÌNH
    # =========================================================================

    # MAE: Sai số tuyệt đối trung bình — đơn vị giống với target ($)
    mae  = mean_absolute_error(y_test, y_pred)

    # RMSE: Căn bậc hai của trung bình bình phương sai số — trừng phạt sai số lớn
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    # R2: Hệ số xác định — nằm trong (-inf, 1.0], giá trị càng gần 1 càng tốt
    r2   = r2_score(y_test, y_pred)

    # In kết quả đánh giá ra terminal
    print("=" * 45)
    print("        KET QUA DANH GIA MO HINH")
    print("=" * 45)
    print(f"  MAE  (Sai so tuyet doi TB): {mae:>10.2f}")
    print(f"  RMSE (Can sai so binh phuong): {rmse:>8.2f}")
    print(f"  R2   (Do phu hop mo hinh): {r2:>11.4f}")
    print("=" * 45)

    # =========================================================================
    # BƯỚC 6: LƯU KẾT QUẢ ĐÁNH GIÁ RA FILE CSV
    # =========================================================================

    # Tạo DataFrame 1 dòng chứa toàn bộ thông tin cần ghi lại
    metrics_df = pd.DataFrame([{
        'model'       : 'RandomForestRegressor',   # Tên thuật toán sử dụng
        'n_estimators': 200,                        # Số lượng cây quyết định
        'test_size'   : 0.2,                        # Tỷ lệ dữ liệu dành cho kiểm tra
        'features'    : str(all_features),          # Danh sách tên đặc trưng đầu vào
        'target'      : target,                     # Tên cột mục tiêu cần dự đoán
        'MAE'         : round(mae,  4),             # Sai số tuyệt đối trung bình
        'RMSE'        : round(rmse, 4),             # Căn sai số bình phương trung bình
        'R2'          : round(r2,   4)              # Hệ số xác định R2
    }])

    # Ghi ra file CSV (không lưu cột index của pandas)
    metrics_df.to_csv(MODEL_METRICS_PATH, index=False)
    print(f"  Metrics da luu tai: {MODEL_METRICS_PATH}")

    # =========================================================================
    # BƯỚC 7: LƯU MÔ HÌNH ĐÃ HUẤN LUYỆN RA FILE .PKL
    # =========================================================================

    # joblib.dump() tuan tu hoa (serialize) toan bo doi tuong Pipeline
    # bao gom ca bo ma hoa (OneHotEncoder) va mo hinh (RandomForest)
    # -> Sau nay chi can goi joblib.load() la dung duoc ngay, khong can train lai
    joblib.dump(pipeline, MODEL_PATH)
    print(f"  Mo hinh da luu tai:  {MODEL_PATH}")

    return pipeline  # Trả về pipeline để caller có thể dùng dự đoán ngay nếu cần