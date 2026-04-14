## 1. Hướng dẫn Cài đặt & Chạy thư viện

Cần cài đặt các thư viện lõi phục vụ phân tích dữ liệu và vẽ biểu đồ. Hãy mở **Terminal** tại thư mục dự án và chạy:

```bash
pip install -r requirements.txt
```
*(Các thư viện chính bao gồm: `pandas`, `numpy`, `matplotlib`, `seaborn`, `scikit-learn`...)*

### Cách chạy toàn bộ Pipeline

Sau khi môi trường đã sẵn sàng, bạn chỉ cần thực thi tập lệnh chính bằng 1 lệnh duy nhất. Bạn có thể nhấn nút **Run/Play** trong IDE (VSCode) hoặc chạy từ command line:

```bash
python src/main.py
```

Quy trình tự động sẽ đi qua các bước:
1. **Load Data**: Đọc file CSV gốc.
2. **Clean Data**: Làm sạch nhiễu tiền tệ (`$`, `,`), chuẩn hóa định dạng ngày tháng, giải quyết các ô trống thiếu dữ liệu.
3. **Feature Engineering**: Tạo thêm đặc trưng mới như Tổng chi phí (`Total Cost`), Tháng du lịch, Phân nhóm tuổi (`Age Group`).
4. **Export Clean Data**: Lưu file dữ liệu sạch mới tinh khôi vào `data/processed/`.
5. **EDA & Visualization**: Vẽ các biểu đồ đẹp mắt và lưu thành ảnh `.png` sẵn sàng để báo cáo.

---

## 2. Cấu trúc Thư mục và Nơi lưu kết quả

Dự án được tổ chức gọn gàng để giảng viên có thể xem luồng dữ liệu hợp logic khoa học:

```text
code/
├── data/
│   ├── raw/                      # Dữ liệu gốc chưa làm sạch
│   │   └── Travel details dataset.csv
│   └── processed/                # DƯ LIỆU ĐÃ LÀM SẠCH (kết quả sau code)
│       └── cleaned_data.csv      <-- Dùng file này nếu cần báo cáo chi tiết
│
├── outputs/
│   └── figures/                  # THƯ MỤC CHỨA BIỂU ĐỒ BÁO CÁO (Kết quả chính)
│       ├── cost_distribution.png # Biểu đồ phân bổ Tổng chi phí các chuyến đi
│       ├── demographics.png      # Biểu đồ giới tính, độ tuổi
│       ├── monthly_trend.png     # Biểu đồ xu hướng du lịch theo từng tháng
│       └── top_destinations.png  # Biểu đồ Top điểm đến yêu thích nhất
│
├── src/                          # MÃ NGUỒN Python
│   ├── analysis/                 # Code vẽ biểu đồ (eda.py)
│   ├── config/                   # Cấu hình chứa mọi đường dẫn
│   ├── ingestion/                # Code lấy dữ liệu
│   ├── preprocessing/            # Code làm sạch (clean.py) và tạo đặc trưng mới
│   └── main.py                   # File trung tâm kích hoạt mọi tác vụ
│
├── requirements.txt              # Danh sách thư viện cần thiết
└── README.md                     # File HDSD này
```
