import os
import platform
import psutil
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
import sys

# Cấu hình giao diện biểu đồ
plt.rcParams.update({'font.size': 11, 'font.family': 'serif'})
sns.set_style("whitegrid")

# Thư mục lưu kết quả báo cáo cuối cùng
OUTPUT_DIR = "./final_paper_artifacts/"
if not os.path.exists(OUTPUT_DIR): os.makedirs(OUTPUT_DIR)

print("🚀 ĐANG KHỞI TẠO HỆ THỐNG TẠO BÁO CÁO TỰ ĐỘNG...\n")

# ==============================================================================
# 1. TỰ ĐỘNG QUÉT CẤU HÌNH MÁY TÍNH (REAL HARDWARE CHECK)
# ==============================================================================
def get_system_info():
    print("🖥️  Đang quét thông tin phần cứng thực tế...")
    try:
        # Lấy thông tin CPU & RAM
        uname = platform.uname()
        ram_bytes = psutil.virtual_memory().total
        ram_gb = round(ram_bytes / (1024 ** 3), 2)
        
        # Lấy thông tin GPU (TensorFlow)
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            try:
                gpu_details = tf.sysconfig.get_build_info()
                cuda_v = gpu_details.get('cuda_version', 'N/A')
                cudnn_v = gpu_details.get('cudnn_version', 'N/A')
                gpu_name = "NVIDIA GPU (Detected via TF)" 
                # Lưu ý: Lấy tên chính xác GPU cần thư viện nvidia-smi hoặc wmi, 
                # ở đây ta dùng thông tin bạn đã cung cấp kết hợp check TF
                gpu_status = f"Available ({len(gpus)} device)"
            except:
                gpu_name = "GPU Detected"
                cuda_v, cudnn_v = "Unknown", "Unknown"
        else:
            gpu_name = "CPU Only"
            cuda_v, cudnn_v = "N/A", "N/A"

        # Tạo DataFrame cho Bảng 3b
        data = {
            "Category": ["Hardware", "Hardware", "Hardware", "Software", "Software", "Software"],
            "Component": ["CPU", "GPU", "RAM", "OS", "Framework", "Python Env"],
            "Specification": [
                f"{uname.processor} ({psutil.cpu_count(logical=True)} threads)", # CPU Thật
                f"{gpu_name} (CUDA {cuda_v}, CuDNN {cudnn_v})",                 # GPU Thật
                f"{ram_gb} GB Total",                                            # RAM Thật
                f"{uname.system} {uname.release} ({uname.machine})",             # OS Thật
                f"TensorFlow {tf.__version__}, Keras {tf.keras.__version__}",    # TF Version Thật
                f"Python {sys.version.split()[0]}"                               # Python Version Thật
            ]
        }
        df = pd.DataFrame(data)
        save_path = os.path.join(OUTPUT_DIR, "Table3b_Real_Environment.csv")
        df.to_csv(save_path, index=False)
        print(f"✅ [Bảng 3b] Đã xuất cấu hình máy thật ra: {save_path}")
        print(df)
        print("-" * 50)
    except Exception as e:
        print(f"❌ Lỗi khi quét phần cứng: {e}")

# ==============================================================================
# 2. LOAD DỮ LIỆU TỪ 4 FILE LOG THỰC TẾ
# ==============================================================================
# Định nghĩa đường dẫn dựa trên code bạn gửi
# ==============================================================================
# 2. LOAD DỮ LIỆU TỪ 4 FILE LOG THỰC TẾ
# ==============================================================================
BASE_PATH = "../../baocao"
FILE_MAP = {
    # [SỬA Ở ĐÂY]: Thêm chữ 'c' vào 'active'
    "CD-AHAL (Proposed)": os.path.join(BASE_PATH, "main_active_learning/plots/history_active_learning.csv"),
    
    "Static Baseline":    os.path.join(BASE_PATH, "main_cnn_gru_attention/plots/history_cnn_gru_attention.csv"),
    "CNN-Attn (No GRU)":  os.path.join(BASE_PATH, "main_cnn_attention/plots/history_cnn_attention_pure.csv"),
    "Weak AL (CNN-Attn)": os.path.join(BASE_PATH, "main_cnn_attention_ative_learning/plots/history_cnn_attention_AL.csv")
}

def load_real_data():
    print("\n📂 Đang đọc dữ liệu log từ các file CSV...")
    dfs = {}
    for name, path in FILE_MAP.items():
        if os.path.exists(path):
            try:
                df = pd.read_csv(path)
                # Chuẩn hóa tên cột về chữ thường để dễ xử lý
                df.columns = [c.lower() for c in df.columns]
                dfs[name] = df
                print(f"   -> Đã tải: {name} ({len(df)} dòng)")
            except Exception as e:
                print(f"   -> ❌ Lỗi đọc file {path}: {e}")
        else:
            print(f"   -> ⚠️ Không tìm thấy file: {path} (Bạn cần chạy code train trước!)")
    return dfs

# ==============================================================================
# 3. TẠO BẢNG ABLATION STUDY TỪ SỐ LIỆU THẬT (TABLE 5)
# ==============================================================================
def generate_real_ablation_table(dfs):
    if not dfs: return
    print("\n📊 Đang tính toán bảng Ablation Study từ dữ liệu thật...")
    
    records = []
    for name, df in dfs.items():
        # Lấy số liệu trung bình (hoặc max/cuối) từ log
        # Giả sử file csv có các cột: accuracy, f1, latency
        
        acc = df['accuracy'].mean() * 100 if 'accuracy' in df.columns else 0
        f1 = df['f1'].mean() * 100 if 'f1' in df.columns else 0
        lat = df['latency'].mean() if 'latency' in df.columns else 0
        
        # Nếu là mô hình Static, lấy 50 batch đầu (trước drift) để công bằng về kiến trúc
        if "Static" in name:
            acc = df['accuracy'].iloc[:50].mean() * 100
        
        records.append({
            "Model Architecture": name,
            "Avg Accuracy (%)": f"{acc:.2f}",
            "Avg F1-Score (%)": f"{f1:.2f}",
            "Avg Latency (ms)": f"{lat:.2f}"
        })
        
    df_table = pd.DataFrame(records)
    save_path = os.path.join(OUTPUT_DIR, "Table5_Real_Ablation.csv")
    df_table.to_csv(save_path, index=False)
    print(f"✅ [Table 5] Đã xuất bảng so sánh số liệu thật: {save_path}")
    print(df_table)

# ==============================================================================
# 4. VẼ BIỂU ĐỒ SO SÁNH THỰC TẾ (FIG 5 & FIG 6b)
# ==============================================================================
def plot_real_comparisons(dfs):
    if not dfs: return
    print("\n📈 Đang vẽ biểu đồ từ dữ liệu thật...")
    
    # --- Hình 5: So sánh Hiệu năng Thích nghi ---
    plt.figure(figsize=(12, 6))
    
    colors = {"CD-AHAL (Proposed)": "#1f77b4", "Static Baseline": "#d62728", 
              "Weak AL (CNN-Attn)": "#ff7f0e", "CNN-Attn (No GRU)": "#7f7f7f"}
    
    for name, df in dfs.items():
        if 'accuracy' in df.columns:
            # Làm mượt dữ liệu (Rolling mean) để biểu đồ đẹp hơn
            y_smooth = df['accuracy'].rolling(window=10, min_periods=1).mean() * 100
            plt.plot(df['batch'], y_smooth, label=name, color=colors.get(name, 'black'), 
                     linewidth=2.5 if "Proposed" in name else 1.5, alpha=0.9)

    plt.title('Real-time Accuracy Comparison (Data from 4 Experiments)')
    plt.xlabel('Streaming Batches')
    plt.ylabel('Accuracy (%)')
    plt.legend(loc='lower right')
    plt.grid(True, linestyle='--', alpha=0.6)
    
    save_path = os.path.join(OUTPUT_DIR, "Fig5_Real_Performance.png")
    plt.savefig(save_path, dpi=300)
    print(f"✅ [Fig 5] Đã vẽ biểu đồ so sánh: {save_path}")

    # --- Hình 6b: So sánh Chiến lược (Dựa trên 2 model AL) ---
    # So sánh giữa CD-AHAL (Strong Arch) và Weak AL (Weak Arch) để thấy tác động
    if "CD-AHAL (Proposed)" in dfs and "Weak AL (CNN-Attn)" in dfs:
        plt.figure(figsize=(10, 6))
        
        df_strong = dfs["CD-AHAL (Proposed)"]
        df_weak = dfs["Weak AL (CNN-Attn)"]
        
        # Chỉ lấy đoạn sau khi Drift (giả sử từ batch 50)
        start_drift = 50
        if len(df_strong) > start_drift and len(df_weak) > start_drift:
            y1 = df_strong['accuracy'].iloc[start_drift:].rolling(5).mean().values * 100
            y2 = df_weak['accuracy'].iloc[start_drift:].rolling(5).mean().values * 100
            
            # Cắt cho bằng độ dài
            min_len = min(len(y1), len(y2))
            x_axis = np.arange(min_len)
            
            plt.plot(x_axis, y1[:min_len], label='CD-AHAL (Strong Arch + AL)', color='blue')
            plt.plot(x_axis, y2[:min_len], label='Weak AL (Weak Arch + AL)', color='orange', linestyle='--')
            
            plt.title('Recovery Speed Comparison (Post-Drift)')
            plt.xlabel('Batches after Drift')
            plt.ylabel('Recovery Accuracy (%)')
            plt.legend()
            plt.grid(True)
            
            save_path_al = os.path.join(OUTPUT_DIR, "Fig6b_Recovery_Speed.png")
            plt.savefig(save_path_al, dpi=300)
            print(f"✅ [Fig 6b] Đã vẽ tốc độ hồi phục: {save_path_al}")

# ==============================================================================
# MAIN RUN
# ==============================================================================
if __name__ == "__main__":
    # 1. Quét máy thật
    get_system_info()
    
    # 2. Đọc dữ liệu thật
    real_dfs = load_real_data()
    
    if real_dfs:
        # 3. Tạo bảng số liệu thật
        generate_real_ablation_table(real_dfs)
        
        # 4. Vẽ biểu đồ thật
        plot_real_comparisons(real_dfs)
        
        print("\n🎉 HOÀN TẤT! Bạn hãy vào thư mục 'final_paper_artifacts' để lấy kết quả.")
    else:
        print("\n⚠️ CẢNH BÁO: Không tìm thấy file log nào.")
        print("   Hãy chắc chắn bạn đã CHẠY 4 file code kia ít nhất 1 lần để sinh ra file .csv")