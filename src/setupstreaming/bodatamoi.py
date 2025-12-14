import pandas as pd
import numpy as np
import os
import glob
from sklearn.utils import shuffle

# --- CẤU HÌNH ---
INPUT_ROOT_DIR = r"D:\DACN\dataset\raw\CICEVSE2024_Dataset\Network Traffic"
# [SỬA 1]: Đổi tên file thành SOURCE (Dữ liệu gốc)
OUTPUT_PATH = r"D:\DACN\dataset\processed\source_data_original.csv" 

ATTACKER_IP = "192.168.137.85"
TARGET_SAMPLES = 1000000 

READ_COLS = [
    'src_ip', 
    'bidirectional_duration_ms', 'src2dst_packets', 'dst2src_packets',
    'bidirectional_mean_piat_ms', 'bidirectional_mean_ps', 'bidirectional_ack_packets',
    # [QUAN TRỌNG]: Cần đọc thêm cột Header Length gốc từ file CSV nếu có.
    # Tuy nhiên, trong CIC-EVSE, Header Length thường phải tự tính hoặc có tên khác.
    # Nếu trong list cột gốc của bạn không có header length, ta vẫn phải để nó.
    # Nhưng nếu bạn muốn Source Data là dữ liệu "Thật", bạn nên giữ giá trị gốc nếu có.
    # Nếu file gốc KHÔNG CÓ cột này, ta buộc phải giữ nguyên logic tính toán hoặc bỏ qua bước gán = 0
]

# Các cột output
FINAL_COLS = [
    "Flow Duration", "Total Fwd Packets", "Total Bwd Packets", 
    "Flow Packets/s", "Flow IAT Mean", "Fwd Header Length", 
    "Packet Length Mean", "ACK Flag Count", "Label"
]

def process_source_data():
    print(f"🚀 [START] Tạo SOURCE DATA từ: {INPUT_ROOT_DIR}")
    
    all_files = glob.glob(os.path.join(INPUT_ROOT_DIR, "**/*.csv"), recursive=True)
    np.random.shuffle(all_files)
    
    benign_dfs = []
    attack_dfs = []
    count_benign = 0
    count_attack = 0

    for file_path in all_files:
        if count_benign >= TARGET_SAMPLES and count_attack >= TARGET_SAMPLES:
            break
            
        try:
            # Check src_ip
            header = pd.read_csv(file_path, nrows=1)
            if 'src_ip' not in header.columns:
                continue

            # Đọc file (Lưu ý: Nếu file gốc có cột header length thì đọc vào, nếu không thì tạm thời để 0 hoặc tính toán)
            # Ở đây mình giả định ta đọc các cột cơ bản
            cols_in_file = [c for c in READ_COLS if c in header.columns]
            df = pd.read_csv(file_path, usecols=cols_in_file)
            
            # 1. Gán nhãn theo IP
            df['Label'] = np.where(df['src_ip'] == ATTACKER_IP, 1, 0)
            
            # 2. Rename & Tính toán
            rename_dict = {
                'bidirectional_duration_ms': 'Flow Duration',
                'src2dst_packets': 'Total Fwd Packets',
                'dst2src_packets': 'Total Bwd Packets',
                'bidirectional_mean_piat_ms': 'Flow IAT Mean',
                'bidirectional_mean_ps': 'Packet Length Mean',
                'bidirectional_ack_packets': 'ACK Flag Count'
            }
            df = df.rename(columns=rename_dict)
            
            df['Flow Duration'] = df['Flow Duration'] * 1000
            df['Flow IAT Mean'] = df['Flow IAT Mean'] * 1000
            total_packets = df['Total Fwd Packets'] + df['Total Bwd Packets']
            duration_s = df['Flow Duration'] / 1e6
            df['Flow Packets/s'] = total_packets / duration_s.replace(0, 1)
            
            # [SỬA 2 - QUAN TRỌNG NHẤT]: XỬ LÝ Fwd Header Length CHO SOURCE DATA
            # Vì file gốc CICEVSE có thể KHÔNG CÓ cột 'Fwd Header Length' sẵn, 
            # chúng ta thường phải tính nó = Total Fwd Packets * 20 (hoặc 32 bytes) tuỳ giao thức.
            # Hoặc nếu bạn muốn Source Data "chuẩn", hãy gán nó một giá trị hợp lý khác 0.
            # Ví dụ: Giả lập Header Length trung bình (thường là 20 bytes/gói TCP cơ bản)
            
            df['Fwd Header Length'] = df['Total Fwd Packets'] * 20 
            # -> Đây là cách ước lượng hợp lý hơn là gán = 0.
            # -> Khi qua file Drift, bạn gán = 0, sự chênh lệch giữa (Packets * 20) và (0) chính là Domain Shift.

            # Lọc cột
            df = df[FINAL_COLS]
            df = df.replace([np.inf, -np.inf], np.nan).fillna(0)

            # 3. Tách Benign/Attack
            df_b = df[df['Label'] == 0]
            df_a = df[df['Label'] == 1]
            
            if not df_b.empty:
                benign_dfs.append(df_b)
                count_benign += len(df_b)
            
            if not df_a.empty and count_attack < TARGET_SAMPLES:
                if len(df_a) > 50000: df_a = df_a.iloc[:50000]
                attack_dfs.append(df_a)
                count_attack += len(df_a)
                
            print(f"⏳ Source Data | Benign: {count_benign} | Attack: {count_attack}", end="\r")

        except Exception:
            continue

    print("\n\n🔄 Đang tổng hợp Source Data...")
    final_benign = pd.concat(benign_dfs, ignore_index=True)
    final_attack = pd.concat(attack_dfs, ignore_index=True)
    
    # Cân bằng
    real_benign_count = len(final_benign)
    if len(final_attack) > real_benign_count:
        final_attack = final_attack.sample(n=real_benign_count, random_state=42)
    
    full_df = pd.concat([final_benign, final_attack], ignore_index=True)
    full_df = shuffle(full_df, random_state=42).reset_index(drop=True)
    
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    full_df.to_csv(OUTPUT_PATH, index=False)
    print(f"🎉 XONG SOURCE DATA! Lưu tại: {OUTPUT_PATH}")
    print(f"📊 Kích thước: {full_df.shape}")

if __name__ == "__main__":
    process_source_data()