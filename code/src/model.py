# =============================================================================
# Module: model.py
# Chức năng: Huấn luyện và so sánh nhiều mô hình học máy để dự đoán chi phí
#            lưu trú của các chuyến đi du lịch. Bao gồm:
#              - So sánh đa mô hình (Ensemble Comparison): RF, XGBoost, Gradient Boosting, LR
#              - Cross-Validation (CV) 5-fold để đánh giá khách quan hơn
#              - Hyperparameter Tuning (GridSearchCV) cho mô hình tốt nhất
#              - Lưu mô hình tốt nhất (best_model.pkl) và toàn bộ metrics
# =============================================================================

import pandas as pd
import numpy as np
import joblib
import warnings
warnings.filterwarnings('ignore')

# --- Thư viện sklearn: Huấn luyện và đánh giá ---
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, KFold
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# --- XGBoost ---
try:
    from xgboost import XGBRegressor
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("  [!] XGBoost chưa được cài đặt. Chạy: pip install xgboost")

from src.config import MODEL_PATH, MODEL_METRICS_PATH, MODEL_PATH_DIR


# =============================================================================
# HÀM PHỤ TRỢ
# =============================================================================

def _build_preprocessor(feature_cat: list) -> ColumnTransformer:
    """Tạo bộ tiền xử lý OneHotEncoder cho các cột Categorical."""
    return ColumnTransformer(transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), feature_cat)
    ], remainder='passthrough')


def _evaluate(y_true, y_pred) -> dict:
    """Tính toán bộ 3 chỉ số đánh giá: MAE, RMSE, R2."""
    mae  = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2   = r2_score(y_true, y_pred)
    return {'MAE': round(mae, 4), 'RMSE': round(rmse, 4), 'R2': round(r2, 4)}


# =============================================================================
# HÀM CHÍNH
# =============================================================================

def train_model(df: pd.DataFrame) -> Pipeline:
    """
    Thực hiện toàn bộ quy trình huấn luyện và so sánh mô hình:
      1. Chọn đặc trưng (Feature Selection)
      2. Tiền xử lý & chia dữ liệu
      3. So sánh đa mô hình có Cross-Validation 5-fold (Ensemble Comparison)
      4. Hyperparameter Tuning (GridSearchCV) cho mô hình tốt nhất
      5. Đánh giá cuối cùng trên tập Test
      6. Lưu mô hình tốt nhất & toàn bộ kết quả so sánh

    Tham số:
        df (pd.DataFrame): DataFrame đã được làm sạch và thêm đặc trưng.

    Trả về:
        Pipeline: Pipeline của mô hình tốt nhất, đã huấn luyện xong.
    """

    # =========================================================================
    # BƯỚC 1: XÁC ĐỊNH ĐẶC TRƯNG VÀ NHÃN MỤC TIÊU
    # =========================================================================
    feature_num = ['Traveler age', 'Duration (days)']
    feature_cat = ['Destination', 'Traveler gender', 'Accommodation type', 'Transportation type']
    target      = 'Accommodation cost'

    # Chỉ giữ lại các cột thực sự tồn tại trong DataFrame
    feature_num = [c for c in feature_num if c in df.columns]
    feature_cat = [c for c in feature_cat if c in df.columns]
    all_features = feature_num + feature_cat

    # =========================================================================
    # BƯỚC 2: CHUẨN BỊ DỮ LIỆU
    # =========================================================================
    df_model = df[all_features + [target]].dropna()
    X = df_model[all_features]
    y = df_model[target]

    # Chia 80/20: train để học, test để đánh giá cuối cùng (Hold-out)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    preprocessor = _build_preprocessor(feature_cat)

    # =========================================================================
    # BƯỚC 3: SO SÁNH ĐA MÔ HÌNH (ENSEMBLE COMPARISON) + CROSS-VALIDATION
    # =========================================================================
    # Danh sách các mô hình ứng cử viên
    candidate_models = {
        'Linear Regression'    : LinearRegression(),
        'Random Forest'        : RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
        'Gradient Boosting'    : GradientBoostingRegressor(n_estimators=100, random_state=42),
    }
    if XGBOOST_AVAILABLE:
        candidate_models['XGBoost'] = XGBRegressor(
            n_estimators=100, random_state=42, n_jobs=-1,
            verbosity=0, eval_metric='rmse'
        )

    # Cross-Validation: KFold 5-fold trên tập TRAIN (không được nhìn vào test)
    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    print("\n" + "=" * 60)
    print("  BUOC 3: SO SANH DA MO HINH (5-Fold Cross-Validation)")
    print("=" * 60)
    print(f"  {'Model':<25} {'CV MAE (mean)':>14} {'CV MAE (std)':>13}")
    print("-" * 60)

    cv_results = {}
    for name, estimator in candidate_models.items():
        pipe = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('model', estimator)
        ])
        # Dùng negative MAE vì sklearn tối thiểu hóa score
        cv_scores = cross_val_score(pipe, X_train, y_train,
                                    cv=kf, scoring='neg_mean_absolute_error', n_jobs=-1)
        mean_mae = -cv_scores.mean()
        std_mae  = cv_scores.std()
        cv_results[name] = {'cv_mae_mean': round(mean_mae, 2), 'cv_mae_std': round(std_mae, 2)}
        print(f"  {name:<25} {mean_mae:>14.2f} {std_mae:>13.2f}")

    print("-" * 60)

    # Chọn mô hình có CV MAE thấp nhất làm best model
    best_model_name = min(cv_results, key=lambda k: cv_results[k]['cv_mae_mean'])
    print(f"\n  => Mo hinh tot nhat (CV MAE nho nhat): [{best_model_name}]")

    # =========================================================================
    # BƯỚC 4: HYPERPARAMETER TUNING (GridSearchCV) CHO MÔ HÌNH TỐT NHẤT
    # =========================================================================
    print("\n" + "=" * 60)
    print(f"  BUOC 4: HYPERPARAMETER TUNING cho [{best_model_name}]")
    print("=" * 60)

    # Định nghĩa không gian tìm kiếm cho từng loại mô hình
    param_grids = {
        'Random Forest': {
            'model__n_estimators': [100, 200, 300],
            'model__max_depth'   : [None, 10, 20],
            'model__min_samples_split': [2, 5],
        },
        'Gradient Boosting': {
            'model__n_estimators'  : [100, 200],
            'model__learning_rate' : [0.05, 0.1, 0.2],
            'model__max_depth'     : [3, 5],
        },
        'XGBoost': {
            'model__n_estimators'  : [100, 200],
            'model__learning_rate' : [0.05, 0.1, 0.2],
            'model__max_depth'     : [3, 5, 7],
        },
        'Linear Regression': {}  # Không có siêu tham số cần tuning
    }

    best_estimator_base = candidate_models[best_model_name]
    best_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('model', best_estimator_base)
    ])

    param_grid = param_grids.get(best_model_name, {})

    if param_grid:
        grid_search = GridSearchCV(
            estimator=best_pipeline,
            param_grid=param_grid,
            cv=kf,
            scoring='neg_mean_absolute_error',
            n_jobs=-1,
            refit=True,          # Tự động fit lại với best params trên toàn bộ X_train
            verbose=0
        )
        grid_search.fit(X_train, y_train)
        best_pipeline = grid_search.best_estimator_
        print(f"  Best params: {grid_search.best_params_}")
        print(f"  Best CV MAE (after tuning): {-grid_search.best_score_:.2f}")
    else:
        # Linear Regression không cần tuning, fit trực tiếp
        best_pipeline.fit(X_train, y_train)
        print(f"  [{best_model_name}] khong co sieu tham so can tuning. Bo qua.")

    # =========================================================================
    # BƯỚC 5: ĐÁNH GIÁ CUỐI CÙNG TRÊN TẬP TEST (Hold-out)
    # =========================================================================
    print("\n" + "=" * 60)
    print("  BUOC 5: DANH GIA CUOI CUNG TREN TAP TEST (Hold-out 20%)")
    print("=" * 60)

    # Đánh giá tất cả các mô hình trên test set để so sánh
    all_metrics_rows = []
    best_test_metrics = {}

    for name, estimator in candidate_models.items():
        pipe_eval = Pipeline(steps=[
            ('preprocessor', _build_preprocessor(feature_cat)),
            ('model', estimator)
        ])
        pipe_eval.fit(X_train, y_train)
        y_pred_eval = pipe_eval.predict(X_test)
        m = _evaluate(y_test, y_pred_eval)
        all_metrics_rows.append({
            'model': name,
            'cv_mae_mean': cv_results[name]['cv_mae_mean'],
            'cv_mae_std' : cv_results[name]['cv_mae_std'],
            **m
        })
        flag = " <== BEST (after CV)" if name == best_model_name else ""
        print(f"  {name:<25} MAE={m['MAE']:>8.2f}  RMSE={m['RMSE']:>8.2f}  R2={m['R2']:>7.4f}{flag}")

    # Lấy metrics của mô hình tốt nhất sau khi tuning
    y_pred_best = best_pipeline.predict(X_test)
    best_test_metrics = _evaluate(y_test, y_pred_best)
    print("-" * 60)
    print(f"  [{best_model_name}] SAU KHI TUNING:")
    print(f"    MAE  = {best_test_metrics['MAE']:.2f}")
    print(f"    RMSE = {best_test_metrics['RMSE']:.2f}")
    print(f"    R2   = {best_test_metrics['R2']:.4f}")
    print("=" * 60)

    # =========================================================================
    # BƯỚC 6: LƯU KẾT QUẢ SO SÁNH & MÔ HÌNH TỐT NHẤT
    # =========================================================================
    # Lưu bảng so sánh tất cả mô hình
    comparison_df = pd.DataFrame(all_metrics_rows)
    comparison_path = MODEL_METRICS_PATH.replace('model_metrics', 'model_comparison')
    comparison_df.to_csv(comparison_path, index=False)
    print(f"\n  Bang so sanh da luu tai: {comparison_path}")

    # Lưu metrics của mô hình tốt nhất (sau tuning) vào file metrics chính
    metrics_df = pd.DataFrame([{
        'model'      : f"{best_model_name} (Tuned)",
        'test_size'  : 0.2,
        'cv_folds'   : 5,
        'features'   : str(all_features),
        'target'     : target,
        **best_test_metrics
    }])
    metrics_df.to_csv(MODEL_METRICS_PATH, index=False)
    print(f"  Metrics mo hinh tot nhat da luu tai: {MODEL_METRICS_PATH}")

    # Lưu mô hình tốt nhất
    joblib.dump(best_pipeline, MODEL_PATH)
    print(f"  Mo hinh tot nhat da luu tai: {MODEL_PATH}")

    # =========================================================================
    # BƯỚC 7: VẼ BIỂU ĐỒ ĐÁNH GIÁ
    # =========================================================================
    from src.visualization import plot_model_evaluation, plot_sample_predictions, plot_model_comparison
    print("\n  Dang ve bieu do danh gia mo hinh...")
    try:
        # Lấy tên đặc trưng sau OneHotEncode
        fitted_preprocessor = best_pipeline.named_steps['preprocessor']
        cat_features_out = fitted_preprocessor.named_transformers_['cat'].get_feature_names_out(feature_cat)
        feature_names = list(cat_features_out) + list(feature_num)

        plot_model_evaluation(y_test, y_pred_best, best_pipeline, feature_names)
        plot_sample_predictions(y_test, y_pred_best, num_samples=15)
        plot_model_comparison(comparison_df)
        print("  Bieu do mo hinh da luu tai: outputs/figures/")
    except Exception as e:
        print(f"  Loi khi ve bieu do: {e}")

    return best_pipeline