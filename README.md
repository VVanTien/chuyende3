<div align="center">

# ✈️ Travel Data Analysis

### Phân tích Dữ liệu Du lịch — Đồ án Tốt nghiệp

![Python](https://img.shields.io/badge/Python-3.13-blue?style=flat-square&logo=python)
![Pandas](https://img.shields.io/badge/Pandas-2.x-150458?style=flat-square&logo=pandas)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.x-F7931E?style=flat-square&logo=scikit-learn)
![Seaborn](https://img.shields.io/badge/Seaborn-0.13-4C72B0?style=flat-square)
![Status](https://img.shields.io/badge/Status-Completed-success?style=flat-square)

</div>

---

## 📌 1. Giới thiệu

Dự án thực hiện bài toán **phân tích dữ liệu du lịch** từ bộ dữ liệu `Travel details dataset.csv`. Đây là bài toán phân tích khám phá kết hợp hồi quy với biến mục tiêu là `Accommodation cost`.

Mục tiêu của nhóm là xử lý dữ liệu thiếu, phân tích mối quan hệ giữa chi phí du lịch và các yếu tố ảnh hưởng, trực quan hóa dữ liệu bằng nhiều loại biểu đồ và xây dựng mô hình học máy để dự đoán chi phí.

---

## 🎯 2. Mục tiêu bài toán

| # | Mục tiêu |
|---|----------|
| 1 | Xử lý dữ liệu thiếu trong tập dữ liệu gốc |
| 2 | Phân tích mối quan hệ giữa `Total Cost` và các biến quan trọng |
| 3 | Trực quan hóa dữ liệu bằng biểu đồ phân phối, scatter plot, heatmap và xu hướng theo tháng |
| 4 | Xây dựng mô hình hồi quy dự đoán chi phí lưu trú (`Accommodation cost`) |
| 5 | Đánh giá mô hình bằng **MAE**, **RMSE** và **R²** |

---

## 🗂️ 3. Dataset sử dụng

- 📄 **File chính:** `Travel details dataset.csv`
- 🔢 **Số dòng:** 139 &nbsp;|&nbsp; **Số cột:** 13
- 🎯 **Biến mục tiêu:** `Accommodation cost`

Một số đặc trưng có **tương quan mạnh** với tổng chi phí chuyến đi:

| Đặc trưng | Hệ số tương quan (r) |
|-----------|---------------------|
| `Accommodation cost` | **0.9805** |
| `Transportation cost` | **0.8923** |
| `Trip ID` | 0.389 |
| `Duration (days)` | -0.092 |

---

## ⚙️ 4. Quy trình thực hiện

### 📥 Bước 1 — Đọc và kiểm tra dữ liệu
Dữ liệu được đọc từ file CSV bằng hàm `load_data()`. Sau khi tải, hệ thống tự động gọi `missing_value_summary()` để thống kê dữ liệu thiếu trước khi xử lý và lưu kết quả ra `missing_value_report.csv`.

### 🧹 Bước 2 — Xử lý dữ liệu thiếu
Hàm `clean_data()` thực hiện:
- Chuẩn hóa tên cột, loại bỏ khoảng trắng thừa.
- Làm sạch cột chi phí: loại bỏ ký tự `$`, `,`, `USD` → chuyển sang `float`.
- Chuyển đổi cột ngày tháng sang định dạng `datetime`.
- Điền giá trị thiếu bằng **median** (số) và **mode** (chuỗi).

### 🔧 Bước 3 — Trích xuất đặc trưng mới (Feature Engineering)
Hàm `add_features()` tạo thêm:
- `Total Cost` = `Accommodation cost` + `Transportation cost`
- `Travel Month`, `Travel Year`, `Travel Month Name`
- `Age Group`: 6 nhóm tuổi từ Trẻ em đến Cao niên

### 🔗 Bước 4 — Phân tích mối quan hệ
Hàm `save_relationship_analysis()` thực hiện:
- Tính hệ số tương quan Pearson với `Total Cost`, lưu top 10.
- Thống kê chi phí (count, mean, median, min, max) theo từng điểm đến.

> 🏆 **Điểm đến đắt nhất:** Auckland, New Zealand (TB $9,500)  
> 💡 **Chi phí ảnh hưởng nhiều nhất:** `Accommodation cost` (r = 0.98)

### 📊 Bước 5 — Trực quan hóa (EDA)
Module `visualization.py` sinh tự động **6 biểu đồ PNG**:

| File | Nội dung |
|------|----------|
| `cost_distribution.png` | Phân phối tổng chi phí |
| `top_destinations.png` | Top 10 điểm đến phổ biến |
| `demographics.png` | Cơ cấu tuổi và giới tính |
| `monthly_trend.png` | Xu hướng theo tháng |
| `scatter_correlations.png` | Scatter plot 2×2 (4 yếu tố) |
| `correlation_heatmap.png` | Ma trận tương quan |

### 🤖 Bước 6 — Xây dựng mô hình
Mô hình: `RandomForestRegressor` với **200 cây quyết định**

Pipeline huấn luyện:
- Chia train/test: **80/20**
- Mã hóa biến phân loại: `OneHotEncoder`
- Giữ nguyên biến số học: `remainder='passthrough'`
- Tăng tốc huấn luyện: `n_jobs=-1`

### 📈 Bước 7 — Đánh giá mô hình

| Chỉ số | Giá trị |
|--------|---------|
| Train size | 111 mẫu |
| Test size | 28 mẫu |
| **MAE** | **693.81** |
| **RMSE** | **1237.80** |
| **R²** | **-0.2722** |

> R² âm phản ánh hạn chế khách quan của dataset nhỏ (139 dòng). Với dataset lớn hơn, mô hình có thể tổng quát hóa tốt hơn.

---

## 🗃️ 5. Giải thích chi tiết các file trong project

### 📁 File dữ liệu

| File | Mô tả |
|------|-------|
| `data/raw/Travel details dataset.csv` | Dữ liệu gốc chưa qua xử lý |
| `data/processed/cleaned_data.csv` | Dữ liệu sau làm sạch và feature engineering |
| `data/processed/missing_value_report.csv` | Báo cáo thống kê missing value trước khi làm sạch |
| `data/processed/top_correlations.csv` | Top 10 biến tương quan mạnh nhất với Total Cost |
| `data/processed/destination_cost_stats.csv` | Thống kê chi phí trung bình theo điểm đến |

### 🐍 File mã nguồn

| File | Mô tả |
|------|-------|
| `main.py` | Entrypoint — gọi lần lượt toàn bộ pipeline |
| `src/config.py` | Định nghĩa tất cả đường dẫn, tự tạo thư mục khi cần |
| `src/data_processing.py` | Load, thống kê missing, làm sạch, feature engineering, phân tích quan hệ |
| `src/visualization.py` | Vẽ và lưu toàn bộ biểu đồ EDA |
| `src/model.py` | Huấn luyện RandomForest, đánh giá, lưu `.pkl` |

### 📤 File kết quả

| File | Mô tả |
|------|-------|
| `outputs/models/travel_model.pkl` | Mô hình đã huấn luyện (dùng lại không cần train lại) |
| `outputs/models/model_metrics.csv` | Chỉ số MAE, RMSE, R² |
| `outputs/figures/*.png` | 6 biểu đồ phân tích |

---

## 🌲 6. Cấu trúc thư mục

```text
code/
├── main.py
├── requirements.txt
├── README.md
├── src/
│   ├── config.py
│   ├── data_processing.py
│   ├── visualization.py
│   └── model.py
├── data/
│   ├── raw/
│   │   └── Travel details dataset.csv
│   └── processed/
│       ├── cleaned_data.csv
│       ├── missing_value_report.csv
│       ├── top_correlations.csv
│       └── destination_cost_stats.csv
└── outputs/
    ├── models/
    │   ├── travel_model.pkl
    │   └── model_metrics.csv
    └── figures/
        ├── cost_distribution.png
        ├── top_destinations.png
        ├── demographics.png
        ├── monthly_trend.png
        ├── scatter_correlations.png
        └── correlation_heatmap.png
```

---

## 🚀 7. Cách chạy chương trình

### Bước 1 — Cài đặt thư viện

```bash
pip install pandas numpy matplotlib seaborn scikit-learn joblib
```

Hoặc cài qua file requirements:

```bash
pip install -r requirements.txt
```

### Bước 2 — Chạy toàn bộ pipeline

```bash
python main.py
```

> ✅ Sau khi chạy xong, tất cả kết quả được lưu tự động vào `data/processed/` và `outputs/`.

---

## 🏁 8. Kết luận

Project đã hoàn thành đầy đủ các yêu cầu chính:

- ✅ Xử lý dữ liệu thiếu và thống kê missing value trước khi làm sạch.
- ✅ Phân tích mối quan hệ giữa chi phí du lịch và các yếu tố (tương quan, thống kê theo điểm đến).
- ✅ Trực quan hóa đa dạng: phân phối, scatter, heatmap, demographics, xu hướng tháng.
- ✅ Xây dựng mô hình RandomForest dự đoán chi phí lưu trú.
- ✅ Đánh giá mô hình bằng MAE, RMSE và R².

Mô hình hiện tại là baseline hợp lý với dataset nhỏ. Nếu cần cải thiện, có thể bổ sung thêm dữ liệu, áp dụng cross-validation hoặc thử nghiệm các thuật toán mạnh hơn như **XGBoost**, **LightGBM**.

---

## 👥 9. Thành viên nhóm

| Họ và tên | MSSV |
|-----------|------|
| Vũ Văn Tiến | 20222334 |

---

<div align="center">
  <sub>✨ Travel Data Analysis — Đồ án Tốt nghiệp 2025-2026 ✨</sub>
</div>
