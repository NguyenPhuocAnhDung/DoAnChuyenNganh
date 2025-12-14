import pandas as pd
import matplotlib.pyplot as plt
import os

# Đường dẫn file (tùy chỉnh lại nếu cần)
REPORT_PATH = "../../baocao/main_cnn_attention/reports"
PLOT_PATH = "../../baocao/main_cnn_attention/plots"
csv_file = os.path.join(REPORT_PATH, "Drift_Events_Detailed.csv")

# Đọc dữ liệu
if os.path.exists(csv_file):
    df = pd.read_csv(csv_file)
    drift_points = df[df['Is_Drift'] == 1]['Batch_ID'].values

    # ==========================================
    # FIGURE 5: ACCURACY OVER TIME
    # ==========================================
    plt.figure(figsize=(12, 6))
    plt.plot(df['Batch_ID'], df['Accuracy'], label='Accuracy (With Active Learning)', color='blue', linewidth=1.5)
    
    # Lưu ý: Để có đường "Without Active Learning", bạn cần chạy lại code nhưng tắt phần model.fit() 
    # ở Giai đoạn 3 và lưu lại log đó để vẽ đè lên. Hiện tại ta chỉ vẽ đường hiện có.
    
    plt.title("Figure 5. Accuracy over time in the data stream")
    plt.xlabel("Batch ID")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_PATH, "Figure_5_Accuracy_Over_Time.png"))
    plt.close()
    print("✅ Đã xuất Figure 5")

    # ==========================================
    # FIGURE 6: UNCERTAINTY CURVE & DRIFT
    # ==========================================
    plt.figure(figsize=(12, 6))
    plt.plot(df['Batch_ID'], df['Uncertainty'], label='Uncertainty', color='red', linewidth=1.5)
    plt.plot(df['Batch_ID'], df['Threshold'], label='Dynamic Threshold', color='orange', linestyle='--', linewidth=1.5)
    
    # Vẽ các vùng Drift (Drift Intervals)
    # Vì drift là 1 điểm, ta vẽ vạch dọc. Nếu muốn "Interval" (khoảng), cần logic xác định độ rộng.
    # Ở đây vẽ vạch dọc tím như code cũ nhưng tách riêng ra.
    for batch_id in drift_points:
        plt.axvline(x=batch_id, color='purple', linestyle='-', alpha=0.5, linewidth=2, label='Drift Detected' if batch_id == drift_points[0] else "")
        
        # Highlight vùng drift (giả sử +/- 2 batch) để giống "Interval" hơn
        plt.axvspan(batch_id - 2, batch_id + 5, color='purple', alpha=0.1)

    plt.title("Figure 6. Uncertainty curve across stream batches highlighting drift intervals")
    plt.xlabel("Batch ID")
    plt.ylabel("Uncertainty Score")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_PATH, "Figure_6_Uncertainty_Drift.png"))
    plt.close()
    print("✅ Đã xuất Figure 6")

else:
    print(f"❌ Không tìm thấy file {csv_file}")