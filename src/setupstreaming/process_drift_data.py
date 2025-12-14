import pandas as pd
import numpy as np
import os
import glob

# --- CẤU HÌNH ---
INPUT_ROOT_DIR = r"D:\DACN\dataset\raw\CICEVSE2024_Dataset\Network Traffic"
OUTPUT_PATH = r"D:\DACN\dataset\processed\drift_test_data_full1.2.csv"

# Đặt số lượng mẫu tối đa muốn lấy. 
# Đặt None nếu muốn lấy hết (cẩn thận RAM), hoặc đặt số cụ thể (ví dụ: 2000000)
MAX_SAMPLES = 2000000 

def process_and_merge_data():
    print(f"🚀 [START] Quét dữ liệu từ: {INPUT_ROOT_DIR}")
    
    all_files = glob.glob(os.path.join(INPUT_ROOT_DIR, "**/*.csv"), recursive=True)
    if not all_files:
        print("❌ Lỗi: Không tìm thấy file .csv nào!")
        return

    print(f"--> Tìm thấy: {len(all_files)} file CSV.")
    
    processed_dfs = []
    total_rows = 0
    
    # Danh sách 8 cột bắt buộc
    required_cols = [
        "Flow Duration", "Total Fwd Packets", "Total Bwd Packets", 
        "Flow Packets/s", "Flow IAT Mean", "Fwd Header Length", 
        "Packet Length Mean", "ACK Flag Count"
    ]

    for file_path in all_files:
        # Kiểm tra giới hạn mẫu
        if MAX_SAMPLES is not None and total_rows >= MAX_SAMPLES:
            print(f"🛑 Đã đạt giới hạn {MAX_SAMPLES} mẫu. Dừng đọc.")
            break

        try:
            print(f"⏳ Đang đọc: {os.path.basename(file_path)}...", end="\r") # In đè dòng để gọn console
            df = pd.read_csv(file_path)

            # 1. Rename
            rename_dict = {
                'bidirectional_duration_ms': 'Flow Duration',
                'src2dst_packets': 'Total Fwd Packets',
                'dst2src_packets': 'Total Bwd Packets',
                'bidirectional_mean_piat_ms': 'Flow IAT Mean',
                'bidirectional_mean_ps': 'Packet Length Mean',
                'bidirectional_ack_packets': 'ACK Flag Count'
            }
            df = df.rename(columns=rename_dict)

            # 2. Kiểm tra sơ bộ
            if 'Flow Duration' not in df.columns:
                print(f"\n   ⚠️ Bỏ qua {os.path.basename(file_path)}: Không đúng định dạng.")
                continue

            # 3. Tính toán & Xử lý đơn vị
            df['Flow Duration'] = df['Flow Duration'] * 1000  # ms -> us
            df['Flow IAT Mean'] = df['Flow IAT Mean'] * 1000  # ms -> us
            
            total_packets = df['Total Fwd Packets'] + df['Total Bwd Packets']
            duration_seconds = df['Flow Duration'] / 1e6
            df['Flow Packets/s'] = total_packets / duration_seconds.replace(0, 1)

            # 4. TẠO DRIFT (QUAN TRỌNG)
            df['Fwd Header Length'] = 0 

            # 5. Lọc cột và làm sạch
            missing = [c for c in required_cols if c not in df.columns]
            if missing:
                print(f"\n   ⚠️ Bỏ qua {os.path.basename(file_path)}: Thiếu cột {missing}")
                continue

            temp_df = df[required_cols].copy()
            temp_df = temp_df.replace([np.inf, -np.inf], np.nan).fillna(0)
            
            processed_dfs.append(temp_df)
            total_rows += len(temp_df)
            
        except Exception as e:
            print(f"\n   ❌ Lỗi file {os.path.basename(file_path)}: {e}")

    if processed_dfs:
        print(f"\n\n🔄 Đang gộp {len(processed_dfs)} DataFrames...")
        final_df = pd.concat(processed_dfs, ignore_index=True)
        
        # Cắt chính xác số lượng mẫu lần cuối nếu lỡ bị thừa do file cuối cùng
        if MAX_SAMPLES is not None and len(final_df) > MAX_SAMPLES:
            final_df = final_df.iloc[:MAX_SAMPLES]
            
        os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
        final_df.to_csv(OUTPUT_PATH, index=False)
        print(f"🎉 XONG! File lưu tại: {OUTPUT_PATH}")
        print(f"📊 Kích thước cuối cùng: {final_df.shape}")
    else:
        print("❌ Không tạo được dữ liệu nào.")

if __name__ == "__main__":
    process_and_merge_data()