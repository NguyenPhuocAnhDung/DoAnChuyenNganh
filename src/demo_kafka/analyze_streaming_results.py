import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

# ==========================================
# CẤU HÌNH
# ==========================================
# Đảm bảo file log này tồn tại (do consumer_active.py sinh ra)
LOG_FILE = "src/active_learning_log.csv" 
OUTPUT_DIR = "../../results/streaming_analysis_plots"

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

# ==========================================
# HÀM VẼ BIỂU ĐỒ
# ==========================================
def analyze_and_plot():
    print(f">>> Đang đọc file log: {LOG_FILE}...")
    
    try:
        # Đọc dữ liệu
        df = pd.read_csv(LOG_FILE)
        
        # Chuyển đổi cột Accuracy về dạng số (nếu cần)
        # Giả sử trong log lưu dạng số thực (0-100)
        
        # Tạo trục X là số thứ tự Batch
        df['Batch'] = range(len(df))
        
        # Tìm các điểm Retraining
        retrain_indices = df[df['Action'].str.contains('RETRAINING', case=False, na=False)].index
        
        print(f"   - Tổng số batch: {len(df)}")
        print(f"   - Số lần học lại (Retraining): {len(retrain_indices)}")
        print(f"   - Độ trễ trung bình: {df['Latency'].mean():.2f} ms")
        
        # --- PLOT 1: Accuracy & Uncertainty Over Time (Chi tiết) ---
        fig, ax1 = plt.subplots(figsize=(14, 7))
        
        color_acc = '#1f77b4'
        ax1.set_xlabel('Batch Sequence', fontsize=12)
        ax1.set_ylabel('Accuracy (%)', color=color_acc, fontsize=12, fontweight='bold')
        ax1.plot(df['Batch'], df['Accuracy'], color=color_acc, linewidth=1.5, label='Accuracy')
        ax1.tick_params(axis='y', labelcolor=color_acc)
        ax1.set_ylim(0, 105)
        ax1.grid(True, alpha=0.3)
        
        # Trục Uncertainty
        ax2 = ax1.twinx()
        color_unc = '#ff7f0e'
        ax2.set_ylabel('Uncertainty', color=color_unc, fontsize=12, fontweight='bold')
        ax2.plot(df['Batch'], df['Uncertainty'], color=color_unc, alpha=0.5, linewidth=1, label='Uncertainty')
        ax2.tick_params(axis='y', labelcolor=color_unc)
        
        # Vẽ vạch đỏ Retrain
        for idx in retrain_indices:
            ax1.axvline(x=idx, color='red', linestyle='--', alpha=0.6, linewidth=0.8)
            
        plt.title('1. Diễn biến Độ chính xác và Độ bất định theo Thời gian thực', fontsize=14)
        
        # Custom Legend
        from matplotlib.lines import Line2D
        custom_lines = [Line2D([0], [0], color=color_acc, lw=2),
                        Line2D([0], [0], color=color_unc, lw=2),
                        Line2D([0], [0], color='red', linestyle='--', lw=1)]
        ax1.legend(custom_lines, ['Accuracy', 'Uncertainty', 'Retraining Point'], loc='lower left')
        
        plt.tight_layout()
        save_path = os.path.join(OUTPUT_DIR, "1_Accuracy_Uncertainty_Trend.png")
        plt.savefig(save_path, dpi=300)
        print(f"✅ Đã lưu: {save_path}")
        plt.close()

        # --- PLOT 2: Phân phối Độ trễ (Latency Histogram) ---
        plt.figure(figsize=(10, 6))
        sns.histplot(df['Latency'], bins=30, kde=True, color='green')
        plt.axvline(x=df['Latency'].mean(), color='red', linestyle='--', label=f"Mean: {df['Latency'].mean():.1f}ms")
        plt.title('2. Phân phối Độ trễ xử lý (Latency Distribution)', fontsize=14)
        plt.xlabel('Latency (ms)')
        plt.ylabel('Frequency (Số lượng Batch)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        save_path = os.path.join(OUTPUT_DIR, "2_Latency_Distribution.png")
        plt.savefig(save_path, dpi=300)
        print(f"✅ Đã lưu: {save_path}")
        plt.close()

        # --- PLOT 3: Số lượng Tấn công phát hiện (Attack Count) ---
        plt.figure(figsize=(12, 5))
        plt.bar(df['Batch'], df['Attacks'], color='#d62728', alpha=0.7, width=1.0)
        plt.title('3. Số lượng Tấn công phát hiện trong mỗi Batch', fontsize=14)
        plt.xlabel('Batch Sequence')
        plt.ylabel('Số gói tin Tấn công')
        plt.grid(True, alpha=0.3, axis='y')
        
        save_path = os.path.join(OUTPUT_DIR, "3_Attack_Detection_Count.png")
        plt.savefig(save_path, dpi=300)
        print(f"✅ Đã lưu: {save_path}")
        plt.close()
        
        # --- PLOT 4: Hiệu quả Phục hồi (Gain Analysis) ---
        # Tính mức tăng Accuracy trung bình sau mỗi lần Retrain (trong khoảng 5 batch tiếp theo)
        gains = []
        for idx in retrain_indices:
            if idx + 5 < len(df):
                acc_before = df['Accuracy'].iloc[idx]
                acc_after = df['Accuracy'].iloc[idx+1:idx+6].mean()
                gains.append(acc_after - acc_before)
        
        if len(gains) > 0:
            plt.figure(figsize=(8, 6))
            plt.boxplot(gains, patch_artist=True, boxprops=dict(facecolor="lightblue"))
            plt.title('4. Mức cải thiện Độ chính xác sau khi Học lại (Accuracy Gain)', fontsize=14)
            plt.ylabel('Mức tăng Accuracy (%)')
            plt.xticks([1], ['Retraining Event'])
            plt.grid(True, alpha=0.3)
            
            save_path = os.path.join(OUTPUT_DIR, "4_Retraining_Gain.png")
            plt.savefig(save_path, dpi=300)
            print(f"✅ Đã lưu: {save_path}")
            plt.close()
            
        print("\n>>> HOÀN TẤT PHÂN TÍCH! Vui lòng kiểm tra thư mục kết quả.")

    except Exception as e:
        print(f"❌ Lỗi khi đọc file log: {e}")
        print("Hãy chắc chắn rằng bạn đã chạy consumer_active.py và có file active_learning_log.csv")

if __name__ == "__main__":
    analyze_and_plot()