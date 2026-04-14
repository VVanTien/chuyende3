import os
import matplotlib.pyplot as plt
import seaborn as sns
from src.config.config import FIGURES_PATH

def set_plot_style():
    """Thiết lập phong cách biểu đồ chung"""
    sns.set_theme(style="whitegrid", palette="muted")
    plt.rcParams.update({'font.size': 12})

def plot_cost_distribution(df):
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

def plot_top_destinations(df):
    """Biểu đồ Top 10 điểm đến phổ biến nhất"""
    if 'Destination' not in df.columns:
        return
        
    plt.figure(figsize=(12, 6))
    top_dest = df['Destination'].value_counts().head(10)
    sns.barplot(y=top_dest.index, x=top_dest.values, palette='viridis')
    plt.title('Top 10 Điểm đến Phổ biến Nhất', fontsize=16, pad=15)
    plt.xlabel('Số lượng chuyến đi', fontsize=14)
    plt.ylabel('Điểm đến', fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_PATH, 'top_destinations.png'), dpi=300)
    plt.close()

def plot_demographics(df):
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

def plot_monthly_trend(df):
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

def run_eda(df):
    """Thực thi toàn bộ pipeline vẽ biểu đồ"""
    print("Đang tiến hành vẽ biểu đồ EDA...")
    set_plot_style()
    plot_cost_distribution(df)
    plot_top_destinations(df)
    plot_demographics(df)
    plot_monthly_trend(df)
    print(f"Hoàn tất vẽ biểu đồ. Các file PNG đã được lưu tại thư mục: {FIGURES_PATH}")
