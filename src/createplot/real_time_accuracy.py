import pandas as pd
import matplotlib.pyplot as plt
import os

# ==============================================================================
# 1. CẤU HÌNH ĐƯỜNG DẪN (QUAN TRỌNG: Hãy sửa đường dẫn tại đây)
# ==============================================================================
# Lưu ý: Thêm chữ r ở trước dấu ngoặc kép để tránh lỗi font đường dẫn Windows
# Ví dụ: r"D:\Result_Folder_1\Drift_Log_CD_AHAL.csv"

FILE_PATHS = {
    "AHAL":    r"D:\DoAnChuyenNganh\baocao\CD_AHAL_FINAL_FULL_METRICS_FINAL\reports_stream\Drift_Log_CD_AHAL.csv",
    
    "CNN_ATT": r"D:\DoAnChuyenNganh\baocao\CNN-Attention-BASELINE\reports_stream\Baseline_CNN_Attention_Stream_Metrics.csv",
    
    "CNN_GRU": r"D:\DoAnChuyenNganh\baocao\CNN-GRU_BASELINE\reports_stream\Baseline_CNN_GRU_Stream_Metrics.csv",
    
    "GRU_ATT": r"D:\DoAnChuyenNganh\baocao\GRU_ATTENTION_BASELINE\reports_stream\Baseline_GRU_Att_Stream_Metrics.csv"
}

# Tên file ảnh kết quả sẽ lưu (lưu tại nơi chạy file script này)
OUTPUT_IMAGE = "final_comparison_result.png"

# ==============================================================================
# 2. HÀM XỬ LÝ
# ==============================================================================
def compare_models_accuracy():
    print("🚀 Đang kiểm tra các file dữ liệu...")
    
    # --- Bước 1: Kiểm tra file tồn tại ---
    valid = True
    for name, path in FILE_PATHS.items():
        if not os.path.exists(path):
            print(f"❌ Lỗi: Không tìm thấy file {name} tại đường dẫn:\n   -> {path}")
            valid = False
        else:
            print(f"✅ Đã tìm thấy: {name}")
    
    if not valid:
        print("\n⚠️ Vui lòng kiểm tra lại đường dẫn trong phần CẤU HÌNH!")
        return

    try:
        # --- Bước 2: Đọc dữ liệu ---
        df_ahal = pd.read_csv(FILE_PATHS["AHAL"])
        df_cnn_att = pd.read_csv(FILE_PATHS["CNN_ATT"])
        df_cnn_gru = pd.read_csv(FILE_PATHS["CNN_GRU"])
        df_gru_att = pd.read_csv(FILE_PATHS["GRU_ATT"])

        # --- Bước 3: Chuẩn hóa tên cột ---
        # CD-AHAL dùng cột 'Recov_Acc', các baseline dùng 'Baseline_Acc'
        df1 = df_ahal[['Batch', 'Recov_Acc']].rename(columns={'Recov_Acc': 'Proposed (CD-AHAL)'})
        df2 = df_cnn_att[['Batch', 'Baseline_Acc']].rename(columns={'Baseline_Acc': 'CNN-Attention'})
        df3 = df_cnn_gru[['Batch', 'Baseline_Acc']].rename(columns={'Baseline_Acc': 'CNN-GRU'})
        df4 = df_gru_att[['Batch', 'Baseline_Acc']].rename(columns={'Baseline_Acc': 'GRU-Attention'})

        # --- Bước 4: Gộp dữ liệu (Inner Join) ---
        # Chỉ so sánh các Batch mà TẤT CẢ model đều có kết quả
        df_final = df1.merge(df2, on='Batch', how='inner') \
                      .merge(df3, on='Batch', how='inner') \
                      .merge(df4, on='Batch', how='inner')

        if df_final.empty:
            print("⚠️ Cảnh báo: Các file không có 'Batch' nào trùng nhau để so sánh.")
            return

        print(f"\n✅ Đang vẽ biểu đồ so sánh trên {len(df_final)} Batch chung...")

        # --- Bước 5: Vẽ biểu đồ ---
        plt.figure(figsize=(14, 7))
        
        # Model Đề xuất (Xanh đậm, Nét to)
        plt.plot(df_final['Batch'], df_final['Proposed (CD-AHAL)'], 
                 label='Proposed (CD-AHAL)', color='#1f77b4', linewidth=3, marker='o', markersize=5)
        
        # Các Model Baseline (Nét đứt)
        plt.plot(df_final['Batch'], df_final['CNN-Attention'], 
                 label='CNN-Attention', color='#ff7f0e', linestyle='--', linewidth=2, alpha=0.8)
        plt.plot(df_final['Batch'], df_final['CNN-GRU'], 
                 label='CNN-GRU', color='#2ca02c', linestyle='--', linewidth=2, alpha=0.8)
        plt.plot(df_final['Batch'], df_final['GRU-Attention'], 
                 label='GRU-Attention', color='#d62728', linestyle='--', linewidth=2, alpha=0.8)

        # Trang trí
        plt.title('Real-time Accuracy Comparison', fontsize=16, fontweight='bold')
        plt.xlabel('Batch Processed', fontsize=12)
        plt.ylabel('Accuracy Score', fontsize=12)
        plt.legend(loc='lower right', fontsize=11, shadow=True)
        plt.grid(True, linestyle=':', alpha=0.6)
        
        # Zoom trục Y để nhìn rõ chênh lệch (từ 0.4 đến 1.0)
        plt.ylim(0.4, 1.05) 

        # --- Bước 6: Lưu và Hiển thị ---
        # Lưu file ảnh tại thư mục hiện tại (nơi chạy script)
        current_dir = os.getcwd()
        save_path = os.path.join(current_dir, OUTPUT_IMAGE)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300)
        print(f"🎉 Xong! Biểu đồ đã lưu tại: {save_path}")
        plt.show()
        
        # In kết quả trung bình ra màn hình
        print("\n--- KẾT QUẢ TRUNG BÌNH (AVG ACC) ---")
        print(df_final.drop('Batch', axis=1).mean().to_string())

    except Exception as e:
        print(f"❌ Có lỗi xảy ra trong quá trình xử lý: {e}")

if __name__ == "__main__":
    compare_models_accuracy()