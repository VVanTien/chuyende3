import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from src.config import FIGURES_PATH

def set_plot_style() -> None:
    """Thiết lập phong cách biểu đồ chuẩn báo cáo chuyên nghiệp"""
    # Sử dụng phong cách thẩm mỹ cao của seaborn
    sns.set_theme(style="whitegrid", context="paper")
    plt.rcParams.update({
        'font.size': 12,
        'axes.titleweight': 'bold',
        'axes.labelweight': 'bold',
        'figure.titleweight': 'bold',
        'figure.titlesize': 18
    })

def plot_missing_values(report: pd.DataFrame) -> None:
    """Biểu đồ phần trăm dữ liệu thiếu (Missing Value)"""
    missing_data = report[report['missing_count'] > 0].copy()
    if missing_data.empty: return
        
    set_plot_style()
    plt.figure(figsize=(12, 6))
    
    ax = sns.barplot(
        data=missing_data, y='column', x='missing_percent (%)', 
        hue='column', palette='Reds_r', legend=False, edgecolor='black'
    )
    
    plt.title('Tỷ lệ Dữ liệu Thiếu (Missing Value) Ban Đầu', fontsize=16, pad=15)
    plt.xlabel('Tỷ lệ thiếu (%)', fontsize=14)
    plt.ylabel('Thuộc tính (Cột)', fontsize=14)
    
    for index, value in enumerate(missing_data['missing_percent (%)']):
        plt.text(value + 0.05, index, f"{value}%", va='center', fontsize=11, fontweight='bold', color='darkred')
        
    plt.xlim(0, missing_data['missing_percent (%)'].max() * 1.2)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_PATH, 'missing_values.png'), dpi=300)
    plt.close()

def plot_cost_distribution(df: pd.DataFrame) -> None:
    """Biểu đồ phân phối chi phí kết hợp (1x2 Subplots)"""
    if 'Total Cost' not in df.columns or 'Cost per day' not in df.columns: return
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Bên trái: Total Cost
    sns.histplot(df['Total Cost'], kde=True, ax=axes[0], color='dodgerblue', edgecolor='black', bins=20)
    axes[0].axvline(df['Total Cost'].mean(), color='red', linestyle='--', linewidth=2.5, label=f"Trung bình: ${df['Total Cost'].mean():.0f}")
    axes[0].axvline(df['Total Cost'].median(), color='green', linestyle='-', linewidth=2.5, label=f"Trung vị: ${df['Total Cost'].median():.0f}")
    axes[0].set_title('Phân phối Tổng Chi Phí Hành Trình (Total Cost)', fontsize=14, pad=10)
    axes[0].set_xlabel('Tổng chi phí ($)')
    axes[0].set_ylabel('Tần suất')
    axes[0].legend(fontsize=11)
    
    # Bên phải: Cost per day
    sns.histplot(df['Cost per day'].dropna(), kde=True, ax=axes[1], color='darkorange', edgecolor='black', bins=20)
    axes[1].axvline(df['Cost per day'].mean(), color='red', linestyle='--', linewidth=2.5, label=f"Trung bình: ${df['Cost per day'].mean():.0f}")
    axes[1].axvline(df['Cost per day'].median(), color='green', linestyle='-', linewidth=2.5, label=f"Trung vị: ${df['Cost per day'].median():.0f}")
    axes[1].set_title('Phân phối Chi Phí Tiêu Tốn Mỗi Ngày (Cost / Day)', fontsize=14, pad=10)
    axes[1].set_xlabel('Chi phí mỗi ngày ($)')
    axes[1].set_ylabel('Tần suất')
    axes[1].legend(fontsize=11)
    
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_PATH, 'cost_distribution.png'), dpi=300)
    plt.close()

def plot_destination_insights(df: pd.DataFrame) -> None:
    """Biểu đồ insights theo điểm đến (Quốc gia) (1x2 Subplots)"""
    if 'Destination Country' not in df.columns: return
    fig, axes = plt.subplots(1, 2, figsize=(18, 7))
    
    # Bên trái: Số lượng khách theo quốc gia
    top_dest = df['Destination Country'].value_counts().head(10)
    sns.barplot(y=top_dest.index, x=top_dest.values, ax=axes[0], palette='viridis', hue=top_dest.index, legend=False, edgecolor='black')
    axes[0].set_title('Top 10 Quốc Gia Đích Đến Phổ Biến Nhất', fontsize=15, pad=10)
    axes[0].set_xlabel('Số lượng chuyến đi')
    axes[0].set_ylabel('')
    for i, v in enumerate(top_dest.values):
        axes[0].text(v + 0.1, i, f" {v}", va='center', fontweight='bold', fontsize=11)
        
    # Bên phải: Chi phí trung bình mỗi ngày đắt nhất
    if 'Cost per day' in df.columns:
        cost_by_dest = df.groupby('Destination Country')['Cost per day'].mean().sort_values(ascending=False).head(10)
        sns.barplot(y=cost_by_dest.index, x=cost_by_dest.values, ax=axes[1], palette='magma', hue=cost_by_dest.index, legend=False, edgecolor='black')
        axes[1].set_title('Top 10 Quốc Gia Đắt Đỏ Nhất (Chi Phí / Ngày)', fontsize=15, pad=10)
        axes[1].set_xlabel('Chi phí trung bình mỗi ngày ($)')
        axes[1].set_ylabel('')
        for i, v in enumerate(cost_by_dest.values):
            axes[1].text(v + 5, i, f" ${v:.0f}", va='center', fontweight='bold', fontsize=11)
            
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_PATH, 'destination_insights.png'), dpi=300)
    plt.close()

def plot_demographics(df: pd.DataFrame) -> None:
    """Biểu đồ nhân khẩu học (Cơ cấu tuổi x Giới tính)"""
    if 'Age Group' not in df.columns or 'Traveler gender' not in df.columns: return
    
    plt.figure(figsize=(10, 6))
    ax = sns.countplot(data=df, x='Age Group', hue='Traveler gender', palette='Set2', edgecolor='black')
    ax.set_title('Cơ cấu Hành khách theo Độ tuổi và Giới tính', fontsize=15, pad=10)
    ax.set_xlabel('Nhóm Tuổi')
    ax.set_ylabel('Số lượng')
    ax.tick_params(axis='x', rotation=25)
    
    # Thêm nhãn số lượng
    for p in ax.patches:
        if p.get_height() > 0:
            ax.annotate(f"{p.get_height():.0f}", (p.get_x() + p.get_width() / 2., p.get_height()), ha='center', va='bottom', fontweight='bold', fontsize=10)
            
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_PATH, 'demographics.png'), dpi=300)
    plt.close()

def plot_age_spending(df: pd.DataFrame) -> None:
    """Biểu đồ phân bố chi tiêu theo độ tuổi (Boxplot + Stripplot)"""
    if 'Age Group' not in df.columns or 'Cost per day' not in df.columns: return
    
    plt.figure(figsize=(10, 6))
    ax = sns.boxplot(data=df, x='Age Group', y='Cost per day', palette='pastel', hue='Age Group', legend=False)
    sns.stripplot(data=df, x='Age Group', y='Cost per day', ax=ax, color='black', alpha=0.3, jitter=True) # Thêm hạt nhiễu để thấy mật độ
    
    ax.set_title('Phân bố Mức độ chịu chi (Chi phí/Ngày) theo Độ tuổi', fontsize=15, pad=10)
    ax.set_xlabel('Nhóm Tuổi')
    ax.set_ylabel('Chi phí mỗi ngày ($)')
    ax.tick_params(axis='x', rotation=25)
        
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_PATH, 'age_spending.png'), dpi=300)
    plt.close()

def plot_temporal_trends(df: pd.DataFrame) -> None:
    """Biểu đồ xu hướng thời gian phức hợp (Trục Y kép và Pie Chart)"""
    if 'Travel Month' not in df.columns: return
    fig = plt.figure(figsize=(16, 12))
    
    # Plot trên: Barplot (Số chuyến) + Lineplot (Chi phí TB)
    ax1 = plt.subplot(2, 1, 1)
    monthly_counts = df['Travel Month'].value_counts().sort_index()
    sns.barplot(x=monthly_counts.index, y=monthly_counts.values, ax=ax1, color='lightblue', edgecolor='black', alpha=0.9, label='Số lượng chuyến')
    ax1.set_ylabel('Số lượng chuyến đi', color='darkblue', fontsize=13)
    ax1.tick_params(axis='y', labelcolor='darkblue')
    
    if 'Total Cost' in df.columns:
        ax2 = ax1.twinx()
        monthly_costs = df.groupby('Travel Month')['Total Cost'].mean().sort_index()
        sns.lineplot(x=range(len(monthly_costs)), y=monthly_costs.values, ax=ax2, color='crimson', marker='D', linewidth=3, markersize=9, label='Chi phí TB')
        ax2.set_ylabel('Chi phí trung bình chuyến đi ($)', color='crimson', fontsize=13)
        ax2.tick_params(axis='y', labelcolor='crimson')
        # Gộp legend của 2 trục
        lines_1, labels_1 = ax1.get_legend_handles_labels()
        lines_2, labels_2 = ax2.get_legend_handles_labels()
        ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper left', fontsize=12)
        
    ax1.set_title('Bức tranh Du lịch theo Tháng: Số lượng và Biến động Chi phí', fontsize=17, pad=15)
    ax1.set_xticks(range(len(monthly_counts)))
    ax1.set_xticklabels([f"Tháng {int(m)}" for m in monthly_counts.index], fontweight='bold')
    ax1.set_xlabel('Tháng trong năm')
    
    # Plot dưới: Pie chart Mùa du lịch
    if 'Travel Season' in df.columns:
        ax3 = plt.subplot(2, 1, 2)
        season_counts = df['Travel Season'].value_counts()
        colors = ['#ff9999','#66b3ff','#99ff99','#ffcc99']
        explode = [0.05] * len(season_counts)
        ax3.pie(season_counts.values, labels=season_counts.index, autopct='%1.1f%%', 
                startangle=140, colors=colors, explode=explode, 
                textprops={'fontsize': 13, 'weight': 'bold'}, shadow=True)
        ax3.set_title('Tỷ trọng Du khách theo Mùa Vụ', fontsize=16, pad=15)
        
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_PATH, 'temporal_trends.png'), dpi=300)
    plt.close()

def plot_cost_breakdown(df: pd.DataFrame) -> None:
    """Biểu đồ cột chồng (Stacked Bar) biểu diễn cấu trúc chi phí"""
    if not all(col in df.columns for col in ['Duration Group', 'Accommodation cost', 'Transportation cost']): return
    
    grouped = df.groupby('Duration Group', observed=False)[['Accommodation cost', 'Transportation cost']].mean()
    
    fig, ax = plt.subplots(figsize=(10, 7))
    grouped.plot(kind='bar', stacked=True, color=['#ff9999','#66b3ff'], ax=ax, edgecolor='black')
    
    ax.set_title('Cấu trúc Chi phí Trung bình theo Độ dài Chuyến đi', fontsize=16, pad=20)
    ax.set_xlabel('Nhóm độ dài chuyến đi', fontsize=14)
    ax.set_ylabel('Chi phí trung bình ($)', fontsize=14)
    plt.xticks(rotation=0, fontweight='bold')
    
    # Chỉnh lại bảng chú giải
    ax.legend(['Lưu trú (Accommodation)', 'Di chuyển (Transportation)'], title='Thành phần chi phí', fontsize=11, title_fontsize=12)
    
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_PATH, 'cost_breakdown.png'), dpi=300)
    plt.close()

def create_scatter_plots(df: pd.DataFrame) -> None:
    """Scatter plot ma trận kèm Đường xu hướng tuyến tính (Regression Line)"""
    features_y = ['Total Cost', 'Cost per day', 'Total Cost', 'Accommodation cost']
    features_x = ['Traveler age', 'Traveler age', 'Duration (days)', 'Transportation cost']
    titles = ['Độ tuổi ảnh hưởng Tổng chi phí?', 'Độ tuổi ảnh hưởng Chi tiêu/Ngày?', 
              'Số ngày đi tác động Tổng chi phí?', 'Tương quan Lưu trú & Di chuyển']
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()
    
    for i, (x_feat, y_feat, title) in enumerate(zip(features_x, features_y, titles)):
        if x_feat in df.columns and y_feat in df.columns:
            # Dùng regplot để tự động tính toán và vẽ đường hồi quy (trend line)
            sns.regplot(data=df, x=x_feat, y=y_feat, ax=axes[i], 
                        scatter_kws={'alpha':0.6, 'color':'teal', 's': 50, 'edgecolor':'white'}, 
                        line_kws={'color':'crimson', 'linewidth':2.5})
            
            axes[i].set_title(title, fontsize=14, pad=10)
            axes[i].set_xlabel(x_feat)
            axes[i].set_ylabel(f'{y_feat} ($)')
            
            # Tính toán và nhúng hệ số tương quan Pearson thẳng vào trong biểu đồ
            corr_val = df[x_feat].corr(df[y_feat])
            box_style = dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.8, edgecolor='gray')
            axes[i].text(0.05, 0.95, f'Tương quan (r): {corr_val:.2f}', 
                         transform=axes[i].transAxes, fontsize=12, fontweight='bold', 
                         verticalalignment='top', bbox=box_style, color='crimson' if abs(corr_val) > 0.5 else 'black')
        
    fig.suptitle('Ma Trận Phân Tán Đa Chiều (Kèm Đường Xu Hướng Tuyến Tính)', fontsize=20, y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES_PATH, 'scatter_correlations.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
def create_correlation_heatmap(df: pd.DataFrame) -> None:
    """Biểu đồ Heatmap biểu diễn ma trận tương quan giữa các biến số"""
    # Lấy các cột giá trị số
    numeric_df = df.select_dtypes(include=['number'])
    
    if not numeric_df.empty and numeric_df.shape[1] > 1:
        plt.figure(figsize=(12, 10))
        correlation_matrix = numeric_df.corr()
        # Tông màu Tươi, Nhẹ, Mướt mắt, Không bị đánh bản quyền và KHÔNG CÓ MÀU ĐỎ: GnBu (Xanh lá mạ -> Xanh biển lơ)
        sns.heatmap(correlation_matrix, annot=True, cmap='GnBu', fmt=".2f", linewidths=0.5, square=True)
        plt.title('Ma Trận Tương Quan Các Đặc Trưng Số', fontsize=16, pad=15)
        plt.tight_layout()
        plt.savefig(os.path.join(FIGURES_PATH, 'correlation_heatmap.png'), dpi=300)
        plt.close()

def plot_model_comparison(comparison_df: pd.DataFrame) -> None:
    """Biểu đồ so sánh hiệu suất tất cả các mô hình (MAE & R² theo từng model)"""
    if comparison_df is None or comparison_df.empty: return
    set_plot_style()

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    models = comparison_df['model'].tolist()
    x = np.arange(len(models))
    colors = sns.color_palette('Set2', len(models))

    # --- Subplot 1: MAE so sánh ---
    bars1 = axes[0].bar(x, comparison_df['MAE'], color=colors, edgecolor='black', width=0.5)
    axes[0].set_title('So sánh MAE (Test Set)\n↓ Càng thấp càng tốt', fontsize=13, fontweight='bold')
    axes[0].set_xticks(x); axes[0].set_xticklabels(models, rotation=20, ha='right', fontsize=11)
    axes[0].set_ylabel('MAE ($)')
    for bar in bars1:
        axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
                     f"{bar.get_height():.1f}", ha='center', va='bottom', fontweight='bold', fontsize=11)

    # --- Subplot 2: RMSE so sánh ---
    bars2 = axes[1].bar(x, comparison_df['RMSE'], color=colors, edgecolor='black', width=0.5)
    axes[1].set_title('So sánh RMSE (Test Set)\n↓ Càng thấp càng tốt', fontsize=13, fontweight='bold')
    axes[1].set_xticks(x); axes[1].set_xticklabels(models, rotation=20, ha='right', fontsize=11)
    axes[1].set_ylabel('RMSE ($)')
    for bar in bars2:
        axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
                     f"{bar.get_height():.1f}", ha='center', va='bottom', fontweight='bold', fontsize=11)

    # --- Subplot 3: CV MAE trung bình (±std) với error bars ---
    axes[2].bar(x, comparison_df['cv_mae_mean'], color=colors, edgecolor='black', width=0.5,
                yerr=comparison_df['cv_mae_std'], capsize=6, error_kw={'linewidth': 2})
    axes[2].set_title('CV MAE Trung bình ± Độ lệch chuẩn\n(5-Fold Cross-Validation)', fontsize=13, fontweight='bold')
    axes[2].set_xticks(x); axes[2].set_xticklabels(models, rotation=20, ha='right', fontsize=11)
    axes[2].set_ylabel('MAE ($)')
    for i, (mean, std) in enumerate(zip(comparison_df['cv_mae_mean'], comparison_df['cv_mae_std'])):
        axes[2].text(i, mean + std + 5, f"{mean:.1f}", ha='center', va='bottom', fontweight='bold', fontsize=11)

    # Đánh dấu mô hình tốt nhất (MAE thấp nhất)
    best_idx = comparison_df['MAE'].idxmin()
    for ax in axes[:2]:
        ax.patches[best_idx].set_edgecolor('gold')
        ax.patches[best_idx].set_linewidth(3)

    fig.suptitle('Kết quả So sánh Tổng thể Các Mô hình Học máy (Ensemble Comparison)',
                 fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_PATH, 'model_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()

def plot_model_evaluation(y_test: pd.Series, y_pred: np.ndarray, pipeline, feature_names: list) -> None:
    """Biểu diễn trực quan chất lượng mô hình: Actual vs Predicted & Feature Importances"""
    set_plot_style()
    fig, axes = plt.subplots(1, 2, figsize=(18, 7))
    
    # Left: Actual vs Predicted
    sns.scatterplot(x=y_test, y=y_pred, ax=axes[0], color='dodgerblue', alpha=0.8, edgecolor='black', s=80)
    
    # Đường x=y (Dự đoán chuẩn xác 100%)
    min_val = min(y_test.min(), y_pred.min())
    max_val = max(y_test.max(), y_pred.max())
    axes[0].plot([min_val, max_val], [min_val, max_val], color='crimson', linestyle='--', linewidth=2.5, label='Đường chuẩn (x=y)')
    
    axes[0].set_title('Thực tế vs. Dự đoán (Actual vs Predicted)', fontsize=15, pad=10)
    axes[0].set_xlabel('Chi phí Thực tế ($)')
    axes[0].set_ylabel('Chi phí Dự đoán ($)')
    axes[0].legend(fontsize=12)

    # Right: Feature Importance
    try:
        # Lấy model RandomForest từ Pipeline
        rf_model = pipeline.named_steps['model']
        importances = rf_model.feature_importances_
        
        # Sắp xếp và chọn top 10
        feat_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
        feat_df = feat_df.sort_values('Importance', ascending=False).head(10)
        
        # Cắt bớt tên đặc trưng nếu quá dài
        feat_df['Feature'] = feat_df['Feature'].apply(lambda x: (x[:25] + '..') if len(x) > 25 else x)
        
        sns.barplot(data=feat_df, x='Importance', y='Feature', hue='Feature', ax=axes[1], palette='viridis', edgecolor='black', legend=False)
        axes[1].set_title('Top 10 Đặc trưng Quan trọng Nhất (Feature Importance)', fontsize=15, pad=10)
        axes[1].set_xlabel('Mức độ ảnh hưởng (Importance)')
        axes[1].set_ylabel('')
        
        # Thêm chỉ số lên chart
        for index, value in enumerate(feat_df['Importance']):
            axes[1].text(value + 0.002, index, f"{value:.3f}", va='center', fontweight='bold', fontsize=11)
            
    except Exception as e:
        print(f"Lỗi vẽ Feature Importance: {e}")
        
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_PATH, 'model_evaluation.png'), dpi=300)
    plt.close()

def plot_sample_predictions(y_test: pd.Series, y_pred: np.ndarray, num_samples: int = 15) -> None:
    """So sánh ngẫu nhiên một vài dự đoán của mô hình với thực tế bằng biểu đồ cột"""
    set_plot_style()
    # Lấy n mẫu đầu tiên
    num_samples = min(num_samples, len(y_test))
    samples_actual = y_test.values[:num_samples]
    samples_pred = y_pred[:num_samples]
    
    x = np.arange(len(samples_actual))
    width = 0.35
    
    plt.figure(figsize=(15, 6))
    plt.bar(x - width/2, samples_actual, width, label='Thực tế', color='dodgerblue', edgecolor='black')
    plt.bar(x + width/2, samples_pred, width, label='Mô hình Dự đoán', color='darkorange', edgecolor='black')
    
    plt.title(f'So sánh Trực tiếp Chi phí Thực Tế vs Dự Đoán ({num_samples} chuyến đi ngẫu nhiên)', fontsize=16, pad=15)
    plt.xlabel('Chuyến đi (Trips)', fontsize=14)
    plt.ylabel('Chi phí Lưu trú ($)', fontsize=14)
    plt.xticks(x, [f'Trip {i+1}' for i in range(len(samples_actual))])
    plt.legend(fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_PATH, 'sample_predictions.png'), dpi=300)
    plt.close()

def plot_executive_dashboard(df: pd.DataFrame) -> None:
    """Dashboard tổng quan phong cách doanh nghiệp (2x2 Grid) - Dựa theo hình mẫu"""
    if 'Travel Year' not in df.columns or 'Travel Month' not in df.columns or 'Total Cost' not in df.columns:
        return
        
    # Chuẩn bị dữ liệu thời gian (loại bỏ NaN trước khi chuyển đổi)
    df_time = df.dropna(subset=['Travel Year', 'Travel Month', 'Total Cost']).copy()
    df_time['YearMonth'] = pd.to_datetime(
        df_time['Travel Year'].astype(int).astype(str) + '-' +
        df_time['Travel Month'].astype(int).astype(str),
        format='%Y-%m'
    )
    df_time = df_time.sort_values('YearMonth')
    
    fig, axes = plt.subplots(2, 2, figsize=(18, 12))
    
    # --- Top Left: Biểu đồ đường có tô nền (Area chart) ---
    ax1 = axes[0, 0]
    time_series = df_time.groupby('YearMonth')['Total Cost'].sum()
    if not time_series.empty:
        ax1.plot(time_series.index, time_series.values, color='blue', linewidth=2.5)
        ax1.fill_between(time_series.index, time_series.values, alpha=0.2, color='steelblue')
        ax1.set_title('Biểu đồ tổng chi phí theo thời gian', fontsize=15, fontweight='bold')
        ax1.set_ylabel('Tổng chi phí ($)', fontsize=13)
        ax1.set_xlabel('Thời gian', fontsize=13)
        ax1.grid(True, linestyle='-', alpha=0.3)
    
    # --- Top Right: Biểu đồ cột (Bar chart) ---
    ax2 = axes[0, 1]
    monthly_avg = df.groupby('Travel Month')['Total Cost'].mean().sort_index()
    if not monthly_avg.empty:
        # Màu sắc lấy cảm hứng từ ảnh mẫu: Xanh lá (Quý 1), Xanh dương (Quý 2,3), Đỏ (Quý 4)
        colors = ['#2ecc71']*3 + ['#3498db']*6 + ['#e74c3c']*3
        colors = (colors * 2)[:len(monthly_avg)]
        
        month_labels = [f"Tháng {int(m)}" for m in monthly_avg.index]
        bars = ax2.bar(month_labels, monthly_avg.values, color=colors, edgecolor='white')
        ax2.set_title('Chi phí trung bình theo tháng', fontsize=15, fontweight='bold')
        ax2.set_ylabel('Chi phí trung bình ($)', fontsize=13)
        ax2.tick_params(axis='x', rotation=45)
        ax2.grid(axis='y', linestyle='-', alpha=0.3)
        
        # Thêm giá trị trên cột
        for bar in bars:
            yval = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2, yval + 10, f"{int(yval)}", ha='center', va='bottom', fontsize=10, fontweight='bold')
            
    # --- Bottom Left: Scatter plot có Trend line ---
    ax3 = axes[1, 0]
    if 'Duration (days)' in df.columns:
        scatter = ax3.scatter(df['Duration (days)'], df['Total Cost'], s=80, alpha=0.7, c=df['Travel Year'], cmap='viridis', edgecolor='white')
        
        # Thêm đường xu hướng (Trend line) đứt đoạn màu đỏ
        valid_idx = df['Duration (days)'].notna() & df['Total Cost'].notna()
        if valid_idx.sum() > 1:
            z = np.polyfit(df.loc[valid_idx, 'Duration (days)'], df.loc[valid_idx, 'Total Cost'], 1)
            p = np.poly1d(z)
            x_trend = np.linspace(df['Duration (days)'].min(), df['Duration (days)'].max(), 100)
            ax3.plot(x_trend, p(x_trend), "r--", linewidth=2.5)
            
        ax3.set_title('Duration vs Total Cost (màu theo năm)', fontsize=15, fontweight='bold')
        ax3.set_ylabel('Tổng chi phí ($)', fontsize=13)
        ax3.set_xlabel('Thời gian lưu trú (ngày)', fontsize=13)
        ax3.grid(True, linestyle='-', alpha=0.3)
        # Thêm colorbar mini cho năm
        plt.colorbar(scatter, ax=ax3, label='Năm')
        
    # --- Bottom Right: Boxplot theo Năm ---
    ax4 = axes[1, 1]
    sns.boxplot(data=df, x='Travel Year', y='Total Cost', ax=ax4, color='white', width=0.4, fliersize=5, linewidth=1.5)
    # Tùy chỉnh màu viền boxplot giống hệt ảnh mẫu
    for i, artist in enumerate(ax4.patches):
        artist.set_edgecolor('#3498db')
        artist.set_facecolor('white')
        
    ax4.set_title('Phân phối chi phí theo năm', fontsize=15, fontweight='bold')
    ax4.set_ylabel('Tổng chi phí ($)', fontsize=13)
    ax4.set_xlabel('Năm', fontsize=13)
    ax4.grid(axis='y', linestyle='-', alpha=0.3)

    fig.suptitle('PHÂN TÍCH TỔNG QUAN CHI PHÍ DU LỊCH', fontsize=22, fontweight='bold', y=0.96)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(os.path.join(FIGURES_PATH, 'executive_dashboard.png'), dpi=300)
    plt.close()

def plot_rolling_trends(df: pd.DataFrame) -> None:
    """Biểu đồ đường với đường trung bình động (Rolling Means) - Dựa theo hình mẫu 2"""
    if 'Travel Year' not in df.columns or 'Travel Month' not in df.columns or 'Total Cost' not in df.columns:
        return
        
    df_time = df.dropna(subset=['Travel Year', 'Travel Month', 'Total Cost']).copy()
    df_time['YearMonth'] = pd.to_datetime(
        df_time['Travel Year'].astype(int).astype(str) + '-' +
        df_time['Travel Month'].astype(int).astype(str),
        format='%Y-%m'
    )
    df_time = df_time.sort_values('YearMonth')
    
    # Tính chi phí trung bình theo tháng
    time_series = df_time.groupby('YearMonth')['Total Cost'].mean()
    if len(time_series) < 4: return
    
    fig, axes = plt.subplots(1, 2, figsize=(18, 6))
    
    # --- Left: Line chart ---
    ax1 = axes[0]
    ax1.plot(time_series.index, time_series.values, marker='o', color='blue', linewidth=2)
    ax1.set_title('Chi phí trung bình theo thời gian', fontsize=15, fontweight='bold')
    ax1.set_ylabel('Chi phí ($)', fontsize=13)
    ax1.set_xlabel('Thời gian', fontsize=13)
    ax1.grid(True, linestyle='-', alpha=0.3)
    
    # --- Right: Line chart with rolling means ---
    ax2 = axes[1]
    ax2.plot(time_series.index, time_series.values, color='blue', alpha=0.4, label='Actual')
    
    # Rolling 3 tháng
    rolling_3 = time_series.rolling(window=3, min_periods=1).mean()
    ax2.plot(time_series.index, rolling_3.values, color='red', linewidth=2, label='Rolling Mean 3 tháng')
    
    # Rolling 6 tháng
    if len(time_series) >= 6:
        rolling_6 = time_series.rolling(window=6, min_periods=1).mean()
        ax2.plot(time_series.index, rolling_6.values, color='green', linewidth=2, label='Rolling Mean 6 tháng')
        
    ax2.set_title('Chi phí với đường trung bình động', fontsize=15, fontweight='bold')
    ax2.set_ylabel('Chi phí ($)', fontsize=13)
    ax2.set_xlabel('Thời gian', fontsize=13)
    ax2.legend()
    ax2.grid(True, linestyle='-', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_PATH, 'rolling_trends.png'), dpi=300)
    plt.close()

def visualization(df: pd.DataFrame) -> None:
    """Thực thi toàn bộ pipeline vẽ biểu đồ chuẩn Report"""
    print("Đang tiến hành vẽ hệ thống biểu đồ EDA chuyên sâu...")
    set_plot_style()
    plot_executive_dashboard(df)
    plot_rolling_trends(df)
    plot_cost_distribution(df)
    plot_destination_insights(df)
    plot_demographics(df)
    plot_age_spending(df)
    plot_temporal_trends(df)
    plot_cost_breakdown(df)
    create_scatter_plots(df)
    create_correlation_heatmap(df)
    print(f"Hoàn tất! Các file PNG phân giải cao (300 DPI) đã được lưu tại: {FIGURES_PATH}")
