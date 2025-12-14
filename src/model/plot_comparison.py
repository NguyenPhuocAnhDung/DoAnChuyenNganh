import matplotlib.pyplot as plt
import numpy as np
import os

# Tạo thư mục lưu nếu chưa có
OUTPUT_DIR = "../../results/comparison_plots"
if not os.path.exists(OUTPUT_DIR): os.makedirs(OUTPUT_DIR)

# ==========================================
# 1. DỮ LIỆU KẾT QUẢ THỰC TẾ
# ==========================================
methods = ['Transformer (SOTA)', 'CNN-GRU (Baseline)', 'Active Learning (Ours)']

# Độ chính xác (Accuracy)
acc_train = [61.89, 99.54, 99.43]      
acc_stream = [20.5, 60.0, 63.0]        

# Tốc độ (Latency - ms)
latency = [176.05, 90.93, 135.50]      

# ==========================================
# 2. VẼ BIỂU ĐỒ SO SÁNH HIỆU NĂNG (Accuracy)
# ==========================================
def plot_accuracy():
    x = np.arange(len(methods))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))
    rects1 = ax.bar(x - width/2, acc_train, width, label='Initial Train Acc', color='#3498db')
    rects2 = ax.bar(x + width/2, acc_stream, width, label='Stream Recovery Acc', color='#e74c3c')

    ax.set_ylabel('Accuracy (%)')
    ax.set_title('So sánh Độ chính xác giữa các phương pháp')
    ax.set_xticks(x)
    ax.set_xticklabels(methods)
    ax.set_ylim(0, 110)
    ax.legend()

    ax.bar_label(rects1, padding=3, fmt='%.1f%%')
    ax.bar_label(rects2, padding=3, fmt='%.1f%%')

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "Comparison_Accuracy.png"))
    print("✅ Đã lưu biểu đồ Accuracy.")
    plt.close()

# ==========================================
# 3. VẼ BIỂU ĐỒ TỐC ĐỘ (Latency)
# ==========================================
def plot_latency():
    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(methods, latency, color=['#95a5a6', '#2ecc71', '#f1c40f'])

    ax.set_ylabel('Latency (ms/batch)')
    ax.set_title('So sánh Tốc độ Xử lý (Thấp hơn là tốt hơn)')
    ax.bar_label(bars, fmt='%.1f ms')

    plt.axhline(y=latency[1], color='green', linestyle='--', alpha=0.5, label='Baseline Speed')
    ax.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "Comparison_Latency.png"))
    print("✅ Đã lưu biểu đồ Latency.")
    plt.close()

# ==========================================
# 4. VẼ BIỂU ĐỒ RADAR (FIXED)
# ==========================================
def plot_radar_summary():
    categories = ['Accuracy', 'Speed (1/Latency)', 'Cost Efficiency', 'Stability']
    N = len(categories)
    
    # 1. Tạo góc (Angles) - Đã fix lỗi dimension
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1] # Đóng vòng tròn (4 góc -> 5 điểm nối)

    # 2. Dữ liệu điểm số (Ước lượng thang 1-10)
    transformer_scores = [2, 4, 5, 2]
    baseline_scores = [9, 10, 1, 8]
    active_scores = [8.5, 8, 10, 9]

    # Đóng vòng tròn dữ liệu
    transformer_scores += transformer_scores[:1]
    baseline_scores += baseline_scores[:1]
    active_scores += active_scores[:1]
    
    # 3. Vẽ
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    
    # Vẽ Transformer
    ax.plot(angles, transformer_scores, label='Transformer (SOTA)', linestyle='dashed')
    ax.fill(angles, transformer_scores, alpha=0.1)
    
    # Vẽ Baseline
    ax.plot(angles, baseline_scores, label='CNN-GRU (Baseline)')
    ax.fill(angles, baseline_scores, alpha=0.1)
    
    # Vẽ Active Learning (Ours)
    ax.plot(angles, active_scores, label='Active Learning (Ours)', linewidth=3, color='red')
    ax.fill(angles, active_scores, alpha=0.2, color='red')
    
    # Chỉnh sửa nhãn trục
    ax.set_xticks(angles[:-1]) # Chỉ lấy 4 góc chính
    ax.set_xticklabels(categories)
    
    ax.set_title('Đánh giá Tổng thể Đa tiêu chí', size=15, y=1.1)
    plt.legend(loc='upper right', bbox_to_anchor=(1.1, 1.1))
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "Comparison_Radar_Overall.png"))
    print("✅ Đã lưu biểu đồ Radar.")
    plt.close()

if __name__ == "__main__":
    print(">>> ĐANG TẠO BIỂU ĐỒ SO SÁNH <<<")
    plot_accuracy()
    plot_latency()
    plot_radar_summary()
    print(f"Xong! Kiểm tra thư mục: {OUTPUT_DIR}")