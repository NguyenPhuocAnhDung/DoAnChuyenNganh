import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np
from matplotlib.gridspec import GridSpec

# ==========================================
# 1. CẤU HÌNH ĐƯỜNG DẪN
# ==========================================
BASE_PATH = "../../baocao"
OUTPUT_PATH = "../../baocao/FINAL_COMPARISON_RESULTS"

if not os.path.exists(OUTPUT_PATH):
    os.makedirs(OUTPUT_PATH)

# Mapping file CSV History
FILES_MAPPING = {
    "main_training/plots/history_general_training.csv": "Baseline (CNN-GRU)",
    "main_ative_learning/reports/final_summary_results.csv": "CD_AHAL",
    # "main_cnn_gru_attention/plots/history_cnn_gru_attention.csv": "CNN-GRU-Attention + AL",
    "main_cnn_attention/plots/history_cnn_attention_pure.csv": "CNN-Attention Pure",
    "main_cnn_attention_ative_learning/plots/history_cnn_attention_AL.csv": "CNN-Attention + AL"
}

# ==========================================
# 2. HÀM LOAD DỮ LIỆU
# ==========================================
def load_data():
    aggregated_df = pd.DataFrame()
    drift_stats = []
    summary_stats = []

    print(f"Checking data in {BASE_PATH}...")

    for rel_path, model_name in FILES_MAPPING.items():
        # 1. Load History Data (Acc, F1, Latency, Uncertainty)
        full_path = os.path.join(BASE_PATH, rel_path)
        
        if os.path.exists(full_path):
            try:
                df = pd.read_csv(full_path)
                df.columns = [c.lower().strip() for c in df.columns] # Chuẩn hóa tên cột
                
                # Fix tên cột
                if 'latency_ms' in df.columns: df = df.rename(columns={'latency_ms': 'latency'})
                if 'unc' not in df.columns and 'uncertainty' in df.columns: df = df.rename(columns={'uncertainty': 'unc'})
                
                # Nếu không có cột unc (ví dụ baseline tĩnh), fill bằng 0
                if 'unc' not in df.columns: df['unc'] = 0.0

                df['Model'] = model_name
                
                # Thống kê cơ bản
                avg_acc = df['accuracy'].mean()
                avg_f1 = df.get('f1', df.get('f1-score', pd.Series([0]*len(df)))).mean()
                avg_lat = df.get('latency', pd.Series([0]*len(df))).mean()
                
                summary_stats.append({
                    'Model': model_name,
                    'Avg_Accuracy': avg_acc,
                    'Avg_F1': avg_f1,
                    'Avg_Latency_ms': avg_lat,
                    'Total_Batches': len(df)
                })
                
                aggregated_df = pd.concat([aggregated_df, df], ignore_index=True)
                print(f"✅ Loaded History: {model_name}")

                # 2. Load Drift Data (Tự động tìm file report tương ứng)
                # Logic: Thay thế 'plots/history_...csv' bằng 'reports/Drift_Events_Detailed.csv'
                folder_part = rel_path.split('/')[0]
                drift_path = os.path.join(BASE_PATH, folder_part, "reports", "Drift_Events_Detailed.csv")
                
                drift_count = 0
                if os.path.exists(drift_path):
                    try:
                        df_drift = pd.read_csv(drift_path)
                        drift_count = len(df_drift)
                    except: pass
                
                drift_stats.append({'Model': model_name, 'Drift_Events': drift_count})

            except Exception as e:
                print(f"❌ Error loading {model_name}: {e}")
        else:
            print(f"⚠️  Missing file: {model_name} (File not found at: {full_path})")

    return aggregated_df, pd.DataFrame(summary_stats), pd.DataFrame(drift_stats)

# ==========================================
# 3. VẼ BIỂU ĐỒ SO SÁNH (MỞ RỘNG)
# ==========================================
def plot_comparisons(df, summary_df, drift_df):
    if df.empty:
        print("No data found. Please run training scripts first.")
        return

    sns.set(style="whitegrid")
    
    # Bảng màu chung
    unique_models = df['Model'].unique()
    palette = sns.color_palette("bright", len(unique_models))
    model_color_map = dict(zip(unique_models, palette))

    # --- FIG A: ACCURACY TIMELINE (Tương tự Figure 5 gốc nhưng so sánh 5 model) ---
    plt.figure(figsize=(14, 7))
    sns.lineplot(data=df, x='batch', y='accuracy', hue='Model', style='Model', alpha=0.9, palette=palette, linewidth=1.5)
    plt.title("So sánh Figure 5: Độ chính xác theo thời gian thực (Real-time Accuracy)", fontsize=14, fontweight='bold')
    plt.xlabel("Batch Index (Time)")
    plt.ylabel("Accuracy")
    plt.legend(bbox_to_anchor=(1.0, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_PATH, "Comparison_Fig5_Accuracy_Timeline.png"))
    plt.close()

    # --- FIG B: UNCERTAINTY MULTI-PANEL (Tương tự Figure 6 gốc) ---
    # Vẽ 5 biểu đồ con xếp chồng lên nhau để so sánh độ bất định
    models = df['Model'].unique()
    fig = plt.figure(figsize=(12, 10))
    gs = GridSpec(len(models), 1, figure=fig, hspace=0.4)

    for i, model in enumerate(models):
        ax = fig.add_subplot(gs[i, 0])
        subset = df[df['Model'] == model]
        
        # Vẽ đường Uncertainty
        ax.plot(subset['batch'], subset['unc'], color=model_color_map[model], label='Uncertainty')
        ax.set_ylabel("Uncertainty")
        ax.set_title(f"Model: {model}", fontsize=10, fontweight='bold', loc='left')
        ax.grid(True, alpha=0.3)
        
        # Nếu model có drift events, đánh dấu vào
        # (Ở đây ta check ngưỡng đơn giản để visualize, hoặc dùng logic 0.001)
        # Để đơn giản, ta tô màu vùng Uncertainty cao
        ax.fill_between(subset['batch'], subset['unc'], 0, color=model_color_map[model], alpha=0.1)

    plt.xlabel("Batch Index")
    plt.suptitle("So sánh Figure 6: Mức độ Bất định (Uncertainty) qua các Model", fontsize=14, fontweight='bold', y=0.95)
    plt.savefig(os.path.join(OUTPUT_PATH, "Comparison_Fig6_Uncertainty_MultiPanel.png"))
    plt.close()

    # --- FIG C: DRIFT EVENTS COUNT (Thống kê số lần thích nghi) ---
    if not drift_df.empty:
        plt.figure(figsize=(10, 6))
        sns.barplot(data=drift_df, x='Model', y='Drift_Events', hue='Model', palette=palette, legend=False)
        plt.title("Tổng số lần phát hiện Drift & Học lại (Retraining Count)", fontsize=14, fontweight='bold')
        plt.ylabel("Số sự kiện Drift")
        plt.xticks(rotation=15)
        
        # Ghi số lên cột
        for p in plt.gca().patches:
            if p.get_height() > 0:
                plt.gca().annotate(f'{int(p.get_height())}', 
                                   (p.get_x() + p.get_width() / 2., p.get_height()), 
                                   ha='center', va='center', xytext=(0, 5), textcoords='offset points')
        
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_PATH, "Comparison_Drift_Counts.png"))
        plt.close()

    # --- FIG D: LATENCY BOXPLOT (Giữ lại cái cũ vì quan trọng) ---
    plt.figure(figsize=(12, 6))
    sns.boxplot(data=df, x='Model', y='latency', hue='Model', legend=False, palette=palette)
    plt.title("Phân phối Độ trễ xử lý (Latency Distribution)", fontsize=14, fontweight='bold')
    plt.ylabel("Latency (ms)")
    plt.xticks(rotation=15)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_PATH, "Comparison_Latency.png"))
    plt.close()
    
    # --- FIG E: PERFORMANCE SUMMARY BAR ---
    melted = summary_df.melt(id_vars=['Model'], value_vars=['Avg_Accuracy', 'Avg_F1'], var_name='Metric', value_name='Score')
    plt.figure(figsize=(12, 6))
    sns.barplot(data=melted, x='Model', y='Score', hue='Metric', palette="viridis")
    plt.title("Tổng hợp Hiệu năng Trung bình (Accuracy vs F1)", fontsize=14, fontweight='bold')
    plt.ylim(0, 1.1)
    plt.xticks(rotation=15)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_PATH, "Comparison_Average_Performance.png"))
    plt.close()

    print(f"\n✅ Đã xuất 5 biểu đồ so sánh chi tiết tại: {OUTPUT_PATH}")

# ==========================================
# 4. MAIN EXECUTION
# ==========================================
if __name__ == "__main__":
    print(">>> STARTING ADVANCED COMPARISON SCRIPT <<<")
    df, summary, drift_stats = load_data()
    
    if not summary.empty:
        # Xuất bảng tổng hợp metrics
        summary.to_csv(os.path.join(OUTPUT_PATH, "Final_Summary_Table.csv"), index=False)
        
        # Xuất bảng tổng hợp drift
        drift_stats.to_csv(os.path.join(OUTPUT_PATH, "Final_Drift_Stats.csv"), index=False)

        print("\n=== FINAL SUMMARY TABLE ===")
        try:
            print(summary.to_markdown(index=False, floatfmt=".4f"))
        except:
            print(summary)
            
        print("\n=== DRIFT EVENTS SUMMARY ===")
        try:
            print(drift_stats.to_markdown(index=False))
        except:
            print(drift_stats)

        plot_comparisons(df, summary, drift_stats)
    else:
        print("❌ Could not generate report due to missing data.")