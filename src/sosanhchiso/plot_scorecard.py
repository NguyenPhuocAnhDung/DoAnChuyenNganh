import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
from math import pi

# ==========================================
# 1. CẤU HÌNH & DỮ LIỆU (Hardcoded Scores)
# ==========================================
OUTPUT_PATH = "../../baocao/FINAL_COMPARISON"
if not os.path.exists(OUTPUT_PATH):
    os.makedirs(OUTPUT_PATH)

# Dữ liệu chấm điểm (Thang 10) dựa trên phân tích trước đó
# Accuracy: Độ chính xác thực tế
# F1-Score: Khả năng phát hiện tấn công
# Speed: Tốc độ xử lý (Điểm cao = Latency thấp)
# Stability: Độ ổn định của biểu đồ Accuracy
# Adaptability: Khả năng phục hồi sau Drift
data = {
    'Model': [
        'Baseline (CNN-GRU)', 
        'Baseline + Active Learning', 
        'CNN-Attention Pure', 
        'CNN-Attention + AL', 
        'CNN-GRU-Attention + AL'
    ],
    'Accuracy':     [6.5, 9.0, 5.0, 6.0, 6.0],
    'F1-Score':     [0.0, 7.0, 9.5, 5.0, 4.5],
    'Speed':        [10.0, 9.0, 6.0, 2.0, 3.0], 
    'Stability':    [8.0, 9.5, 4.0, 5.0, 5.0],  
    'Adaptability': [2.0, 9.0, 5.0, 7.0, 6.0]   
}

df = pd.DataFrame(data)

# ==========================================
# 2. HÀM VẼ RADAR CHART (5 MODEL CHUNG 1 HÌNH)
# ==========================================
def plot_master_radar():
    # Các chỉ số
    categories = list(df.columns[1:])
    N = len(categories)

    # Góc vẽ cho từng trục
    angles = [n / float(N) * 2 * pi for n in range(N)]
    angles += angles[:1] # Khép kín vòng tròn

    # Khởi tạo hình
    plt.figure(figsize=(10, 10))
    ax = plt.subplot(111, polar=True)

    # Vẽ trục & Nhãn
    plt.xticks(angles[:-1], categories, color='black', size=12, fontweight='bold')
    ax.set_rlabel_position(0)
    plt.yticks([2, 4, 6, 8, 10], ["2", "4", "6", "8", "10"], color="grey", size=10)
    plt.ylim(0, 10)

    # Màu sắc cho 5 model
    colors = ['#7f7f7f', '#d62728', '#1f77b4', '#2ca02c', '#ff7f0e']
    
    # Vẽ từng model lên chung 1 biểu đồ
    for i, row in df.iterrows():
        values = row.drop('Model').values.flatten().tolist()
        values += values[:1] # Khép kín
        
        ax.plot(angles, values, linewidth=2, linestyle='solid', label=row['Model'], color=colors[i])
        ax.fill(angles, values, color=colors[i], alpha=0.05) # Tô màu mờ bên trong

    # Trang trí
    plt.title('So sánh Tổng quan 5 Mô hình trên 5 Chỉ số', size=16, weight='bold', y=1.08)
    plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), title="Danh sách Mô hình")

    # Lưu ảnh
    save_path = os.path.join(OUTPUT_PATH, "Master_Radar_5_Metrics.png")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"✅ Đã xuất biểu đồ Radar tổng hợp tại: {save_path}")

# ==========================================
# 3. MAIN
# ==========================================
if __name__ == "__main__":
    plot_master_radar()