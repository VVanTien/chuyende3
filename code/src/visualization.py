import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from src.config import FIGURES_PATH

def set_plot_style() -> None:
    """Thiết lập phong cách biểu đồ chung"""
    sns.set_theme(style="whitegrid", palette="muted")
    plt.rcParams.update({'font.size': 12})

def plot_cost_distribution(df: pd.DataFrame) -> None:
    """Biểu đồ phân phối tổng chi phí chuyến đi"""
    if 'Total Cost' not in df.columns:
        return
    
    plt.figure(figsize=(10, 6))
    sns.histplot(df['Total Cost'], bins=20, kde=True, color='skyblue')
    plt.title('Phân phối Tổng chi phí Du lịch', fontsize=16, pad=15)
    plt.xlabel('Tổng chi phí ($)', fontsize=14)
    plt.ylabel('Số lượng chuyến đi', fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_PATH, 'cost_distribution.png'), dpi=300)
    plt.close()

def plot_top_destinations(df: pd.DataFrame) -> None:
    """Biểu đồ Top 10 điểm đến phổ biến nhất"""
    if 'Destination' not in df.columns:
        return
        
    plt.figure(figsize=(12, 6))
    top_dest = df['Destination'].value_counts().head(10)
    sns.barplot(y=top_dest.index, x=top_dest.values, hue=top_dest.index, palette='viridis', legend=False)
    plt.title('Top 10 Điểm đến Phổ biến Nhất', fontsize=16, pad=15)
    plt.xlabel('Số lượng chuyến đi', fontsize=14)
    plt.ylabel('Điểm đến', fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_PATH, 'top_destinations.png'), dpi=300)
    plt.close()

def plot_demographics(df: pd.DataFrame) -> None:
    """Biểu đồ nhóm tuổi và giới tính"""
    if 'Age Group' in df.columns and 'Traveler gender' in df.columns:
        plt.figure(figsize=(10, 6))
        sns.countplot(data=df, x='Age Group', hue='Traveler gender', palette='pastel')
        plt.title('Phân bố Độ tuổi theo Giới tính', fontsize=16, pad=15)
        plt.xlabel('Nhóm Tuổi', fontsize=14)
        plt.ylabel('Số lượng', fontsize=14)
        plt.xticks(rotation=45)
        plt.legend(title='Giới tính')
        plt.tight_layout()
        plt.savefig(os.path.join(FIGURES_PATH, 'demographics.png'), dpi=300)
        plt.close()

def plot_monthly_trend(df: pd.DataFrame) -> None:
    """Biểu đồ xu hướng du lịch theo tháng"""
    if 'Travel Month' in df.columns:
        plt.figure(figsize=(10, 6))
        monthly_counts = df['Travel Month'].value_counts().sort_index()
        sns.lineplot(x=monthly_counts.index, y=monthly_counts.values, marker='o', color='coral', linewidth=2.5)
        plt.title('Xu hướng Du lịch theo Tháng', fontsize=16, pad=15)
        plt.xlabel('Tháng', fontsize=14)
        plt.ylabel('Số lượng chuyến đi', fontsize=14)
        plt.xticks(range(1, 13))
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(FIGURES_PATH, 'monthly_trend.png'), dpi=300)
        plt.close()

def create_scatter_plots(df: pd.DataFrame) -> None:
    """Biểu đồ Scatter plot biểu diễn nhiều tương quan trong cùng 1 khung ảnh bằng vòng lặp."""
    # 4 yếu tố chính quyết định trực tiếp tới tổng chi phí chuyến đi
    features = ['Traveler age', 'Duration (days)', 'Accommodation cost', 'Transportation cost']
    
    if 'Total Cost' not in df.columns:
        return

    # Khởi tạo khung hiển thị 2 hàng 2 cột
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    for ax, feature in zip(axes, features):
        if feature in df.columns:
            # Phân tách màu theo giới tính nếu khả dụng
            if 'Traveler gender' in df.columns:
                sns.scatterplot(
                    data=df, x=feature, y='Total Cost', hue='Traveler gender', 
                    alpha=0.65, s=60, ax=ax
                )
            else:
                sns.scatterplot(
                    data=df, x=feature, y='Total Cost', 
                    alpha=0.65, s=60, ax=ax, color="#1f77b4"
                )
            
            # Cân chỉnh thông số thẩm mỹ
            ax.set_title(f"Tổng Chi Phí vs {feature}", fontsize=13)
            ax.set_xlabel(feature, fontsize=11)
            ax.set_ylabel('Tổng Chi Phí ($)', fontsize=11)
            # Tắt hiển thị số thực khoa học trên trục y
            ax.ticklabel_format(style="plain", axis="y")
        else:
            # Ẩn ô hình đó nếu dữ liệu tính toán bị thiếu
            ax.set_visible(False)

    fig.suptitle('Mối Tương Quan Toàn Diện Giữa Các Yếu Tố Và Tổng Chi Phí Du Lịch', fontsize=18, y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES_PATH, 'scatter_correlations.png'), dpi=200, bbox_inches='tight')
    plt.close(fig)

def create_correlation_heatmap(df: pd.DataFrame) -> None:
    """Biểu đồ Heatmap biểu diễn ma trận tương quan giữa các biến số"""
    # Lấy các cột giá trị số
    numeric_df = df.select_dtypes(include=['number'])
    
    if not numeric_df.empty and numeric_df.shape[1] > 1:
        plt.figure(figsize=(10, 8))
        correlation_matrix = numeric_df.corr()
        # Tông màu Tươi, Nhẹ, Mướt mắt, Không bị đánh bản quyền và KHÔNG CÓ MÀU ĐỎ: GnBu (Xanh lá mạ -> Xanh biển lơ)
        sns.heatmap(correlation_matrix, annot=True, cmap='GnBu', fmt=".2f", linewidths=0.5, square=True)
        plt.title('Ma Trận Tương Quan Các Đặc Trưng Số', fontsize=16, pad=15)
        plt.tight_layout()
        plt.savefig(os.path.join(FIGURES_PATH, 'correlation_heatmap.png'), dpi=300)
        plt.close()

def visualization(df: pd.DataFrame) -> None:
    """Thực thi toàn bộ pipeline vẽ biểu đồ"""
    print("Đang tiến hành vẽ biểu đồ EDA...")
    set_plot_style()
    plot_cost_distribution(df)
    plot_top_destinations(df)
    plot_demographics(df)
    plot_monthly_trend(df)
    create_scatter_plots(df)
    create_correlation_heatmap(df)
    print(f"Hoàn tất vẽ biểu đồ. Các file PNG đã được lưu tại thư mục: {FIGURES_PATH}")
