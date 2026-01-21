import pandas as pd
import numpy as np
import os
import glob
import joblib
from sklearn.utils import shuffle, resample
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from collections import Counter

# ==========================================
# 1. CẤU HÌNH HỆ THỐNG
# ==========================================
BASE_DATASET_PATH = "D:/DACN/dataset/raw"

DIRS = {
    "CIC2017":  (os.path.join(BASE_DATASET_PATH, "CICDDoS2017"), "parquet", True),
    "IoT2023":  (os.path.join(BASE_DATASET_PATH, "CICIoT2023"), "csv", True),
    "CIC2018":  (os.path.join(BASE_DATASET_PATH, "CSE-CIC-IDS-2018"), "csv", True),
    "DDoS2019": (os.path.join(BASE_DATASET_PATH, "CICDDoS2019"), "parquet", True),
    "Darknet":  (os.path.join(BASE_DATASET_PATH, "CICDarknet2020CSVs"), "csv", True),
    "DoH":      (os.path.join(BASE_DATASET_PATH, "L1-DoH-NonDoH.parquet"), "parquet", False) 
}

# [OUTPUT] Folder lưu trữ kết quả cuối cùng
OUTPUT_PATH = "../../dataset/processed_v3_unified"
if not os.path.exists(OUTPUT_PATH): os.makedirs(OUTPUT_PATH)

TARGET_COLS = [
    'Flow Duration', 'Total Fwd Packets', 'Total Bwd Packets', 
    'Fwd Packet Length Max', 'Fwd Packet Length Min', 'Fwd Packet Length Mean', 
    'Bwd Packet Length Max', 'Bwd Packet Length Min', 'Bwd Packet Length Mean', 
    'Flow Packets/s', 'Flow IAT Mean', 'Flow IAT Std', 'Flow IAT Max', 'Flow IAT Min',
    'Fwd IAT Mean', 'Bwd IAT Mean', 'Fwd Header Length', 'Bwd Header Length', 
    'Fwd Packets/s', 'Bwd Packets/s', 'Min Packet Length', 'Max Packet Length', 
    'Packet Length Mean', 'Packet Length Std', 'FIN Flag Count', 'SYN Flag Count', 
    'RST Flag Count', 'PSH Flag Count', 'ACK Flag Count', 'URG Flag Count', 
    'Average Packet Size', 'Init_Win_bytes_forward', 'Init_Win_bytes_backward', 
    'Active Mean', 'Idle Mean', 'Label'
]

# ==========================================
# 2. HÀM MAPPING & CLEANING
# ==========================================
COLUMN_MAP_DICT = {
    'flow_duration': 'Flow Duration', 'duration': 'Flow Duration', 
    'tot_fwd_pkts': 'Total Fwd Packets', 'total_fwd_packets': 'Total Fwd Packets',
    'tot_bwd_pkts': 'Total Bwd Packets', 'total_bwd_packets': 'Total Bwd Packets',
    'flow_pkts_s': 'Flow Packets/s', 'rate': 'Flow Packets/s',
    'flow_iat_mean': 'Flow IAT Mean',
    'fwd_header_len': 'Fwd Header Length', 'header_length': 'Fwd Header Length',
    'protocol_type': 'Protocol', 'protocol': 'Protocol',
    'label': 'Label',
    'fin_flag_number': 'FIN Flag Count', 'syn_flag_number': 'SYN Flag Count',
    'rst_flag_number': 'RST Flag Count', 'psh_flag_number': 'PSH Flag Count',
    'ack_flag_number': 'ACK Flag Count', 'urg_flag_number': 'ACK Flag Count', 
    'ece_flag_number': 'ECE Flag Count', 'cwe_flag_count': 'CWE Flag Count',
    'max': 'Max Packet Length', 'min': 'Min Packet Length', 
    'mean': 'Packet Length Mean', 'std': 'Packet Length Std',
    'Duration': 'Flow Duration',
    'FlowBytesSent': 'Total Length of Fwd Packets',
    'FlowBytesReceived': 'Total Length of Bwd Packets',
    'FlowSentRate': 'Fwd Packets/s',
    'FlowReceivedRate': 'Bwd Packets/s',
    'PacketLengthMean': 'Packet Length Mean',
    'PacketLengthStandardDeviation': 'Packet Length Std',
    'PacketLengthVariance': 'Packet Length Variance',
    'DoH': 'Label',
    'flow.duration': 'Flow Duration', 
    'total.fwd.packets': 'Total Fwd Packets', 
    'total.bwd.packets': 'Total Bwd Packets', 
    'flow.packets.s': 'Flow Packets/s', 
    'flow.iat.mean': 'Flow IAT Mean',
    'label': 'Label', 'traffic category': 'Label'
}

def normalize_columns(df):
    new_cols = []
    for col in df.columns:
        c_clean = str(col).strip()
        c_lower = c_clean.lower()
        final_name = col 
        if c_clean in COLUMN_MAP_DICT: final_name = COLUMN_MAP_DICT[c_clean]
        elif c_lower in COLUMN_MAP_DICT: final_name = COLUMN_MAP_DICT[c_lower]
        else:
            cl = c_lower.replace('_', '').replace('.', '').replace(' ', '')
            if 'flow' in cl and 'duration' in cl: final_name = 'Flow Duration'
            elif 'tot' in cl and 'fwd' in cl and 'pkt' in cl: final_name = 'Total Fwd Packets'
            elif 'tot' in cl and 'bwd' in cl and 'pkt' in cl: final_name = 'Total Bwd Packets'
            elif 'flow' in cl and 'pkt' in cl and 's' in cl: final_name = 'Flow Packets/s'
            elif 'iat' in cl and 'mean' in cl and 'flow' in cl: final_name = 'Flow IAT Mean'
            elif 'label' in cl or 'class' in cl or 'category' in cl: final_name = 'Label'
        new_cols.append(final_name)
    df.columns = new_cols
    return df.loc[:, ~df.columns.duplicated()]

def standardize_label(label):
    lbl = str(label).lower().strip()
    if lbl == 'true' or lbl == 'doh': return 7
    if lbl == 'false' or lbl == 'nondoh': return 0
    if lbl in ['benign', 'normal', '0', '0.0', 'non-tor']: return 0
    if 'dos' in lbl and 'ddos' not in lbl: return 1 
    if 'port' in lbl or 'scan' in lbl or 'nmap' in lbl: return 2 
    if 'bot' in lbl or 'mirai' in lbl: return 3           
    if 'brute' in lbl or 'ssh' in lbl or 'ftp' in lbl: return 4 
    if 'web' in lbl or 'xss' in lbl or 'sql' in lbl: return 5   
    if 'infil' in lbl or 'backdoor' in lbl: return 6      
    if 'ddos' in lbl or 'udp' in lbl or 'tcp' in lbl: return 7   
    if 'tor' in lbl or 'vpn' in lbl: return 7               
    if 'malicious' in lbl: return 7       
    return 7 

def clean_data(df):
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 0:
        medians = df[numeric_cols].median()
        df.fillna(medians, inplace=True)
    df.fillna(0, inplace=True)
    return df

# --- [NEW] Hàm in thông tin chi tiết ---
def print_dataset_stats(df, name="Dataset"):
    print(f"\n📊 [STATS] {name}")
    if df.empty:
        print("   ❌ Empty DataFrame")
        return
    
    print(f"   ► Rows (Samples): {len(df):,}")
    print(f"   ► Columns: {df.shape[1]}")
    
    if 'Label_Multi' in df.columns:
        counts = df['Label_Multi'].value_counts().sort_index()
        print(f"   ► Label Distribution (Multi):")
        for lbl, count in counts.items():
            print(f"      - Label {lbl}: {count:,}")
    else:
        print("   ⚠️ Column 'Label_Multi' not found.")
    print("-" * 30)

def analyze_quality(df, name="Dataset"):
    print(f"\n🔍 [CHECK] PHÂN TÍCH CHẤT LƯỢNG: {name}")
    print("-" * 40)
    total_rows = len(df)
    unique_rows = len(df.drop_duplicates())
    dup_count = total_rows - unique_rows
    dup_percent = (dup_count / total_rows) * 100 if total_rows > 0 else 0
    print(f"1. Tổng mẫu: {total_rows}")
    print(f"   - Mẫu duy nhất: {unique_rows} | Trùng lặp: {dup_count} ({dup_percent:.2f}%)")
    
    if 'Label_Multi' in df.columns:
        print(f"3. Chi tiết (Label_Multi):")
        print(df['Label_Multi'].value_counts().sort_index())
    print("-" * 40)

# ==========================================
# 3. HÀM CỐT LÕI (SMOTE & READ)
# ==========================================
def smart_smote_balance(df, target_max=300000, target_min=50000):
    if df.empty: return df
    y = df['Label_Multi']
    X = df.drop(columns=['Label', 'Label_Multi', 'Label_Bin'], errors='ignore')
    
    counter = Counter(y)
    df_temp = df.copy()
    undersampled_dfs = []
    classes_to_smote = []
    
    # 1. Undersample lớp lớn
    for cls, count in counter.items():
        df_cls = df_temp[df_temp['Label_Multi'] == cls]
        if count > target_max:
            df_res = resample(df_cls, replace=False, n_samples=target_max, random_state=42)
            undersampled_dfs.append(df_res)
        else:
            undersampled_dfs.append(df_cls)
            classes_to_smote.append(cls)
            
    df_reduced = pd.concat(undersampled_dfs)
    X_red = df_reduced.drop(columns=['Label', 'Label_Multi', 'Label_Bin'], errors='ignore')
    y_red = df_reduced['Label_Multi']
    
    # 2. SMOTE lớp nhỏ
    smote_dict = {}
    curr_counts = Counter(y_red)
    
    for cls in classes_to_smote:
        if curr_counts[cls] < 6: continue # Quá ít để SMOTE
        if curr_counts[cls] < target_min: smote_dict[cls] = target_min
            
    if not smote_dict: return df_reduced
        
    print(f"   -> Chạy SMOTE cho các lớp: {smote_dict.keys()}...")
    try:
        smote = SMOTE(sampling_strategy=smote_dict, random_state=42, k_neighbors=5)
        X_res, y_res = smote.fit_resample(X_red, y_red)
        
        df_final = pd.DataFrame(X_res, columns=X.columns)
        df_final['Label_Multi'] = y_res
        df_final['Label_Bin'] = df_final['Label_Multi'].apply(lambda x: 0 if x==0 else 1)
        df_final['Label'] = df_final['Label_Bin']
        return df_final
    except Exception as e:
        print(f"   ❌ SMOTE Lỗi: {e}. Giữ nguyên.")
        return df_reduced

def read_dataset_optimized(key, limit):
    path_info = DIRS.get(key)
    if not path_info: return pd.DataFrame()
    path, ext, recursive = path_info
    
    if os.path.isfile(path): files = [path]
    else:
        if recursive: files = glob.glob(os.path.join(path, "**", f"*.{ext}"), recursive=True)
        else: files = glob.glob(os.path.join(path, f"*.{ext}"))

    print(f"--> Đang xử lý: {key} (Tìm thấy {len(files)} files)")
    df_list = []
    max_read = 300 if key in ["IoT2023", "Darknet"] else 80
    count_ok = 0
    
    for i, f in enumerate(files):
        if count_ok >= 40 or i > max_read: break 
        try:
            if f.endswith('.parquet'): temp = pd.read_parquet(f)
            else: temp = pd.read_csv(f, encoding='latin1', on_bad_lines='skip', low_memory=False)
            
            temp = normalize_columns(temp)
            cols_present = [c for c in TARGET_COLS if c in temp.columns]
            if len(cols_present) < 2: continue
            temp = temp[cols_present]
            if 'Label' not in temp.columns: continue
            
            temp['Label_Multi'] = temp['Label'].apply(standardize_label)
            temp['Label_Bin'] = temp['Label_Multi'].apply(lambda x: 0 if x==0 else 1)
            
            # Lấy mẫu
            if len(temp) > limit: temp = temp.sample(n=limit, random_state=42)
            
            df_list.append(temp)
            count_ok += 1
        except Exception as e: pass

    if not df_list: 
        print(f"   ⚠️ {key}: Không đọc được dữ liệu nào.")
        return pd.DataFrame()
    
    # --- [MODIFIED] Tổng hợp và in Info ---
    final_df = pd.concat(df_list)
    print_dataset_stats(final_df, key) # <--- Gọi hàm in info tại đây
    return final_df

# ==========================================
# 4. PIPELINE CHÍNH (ALL-IN-ONE)
# ==========================================
def main():
    print("--- BẮT ĐẦU PIPELINE XỬ LÝ DỮ LIỆU TỔNG HỢP (V3.0) ---")
    
    # ---------------------------------------------------------
    # BƯỚC 1: ĐỌC DỮ LIỆU GỐC CHO TRAIN
    # ---------------------------------------------------------
    print("\n[PHASE 1] GOM DỮ LIỆU TRAIN (CIC2017 + IoT2023)")
    df_2017 = read_dataset_optimized("CIC2017", 100000) 
    df_iot  = read_dataset_optimized("IoT2023", 100000) 
    
    if df_2017.empty and df_iot.empty: print("Lỗi: Không có dữ liệu Train"); return

    df_train_raw = pd.concat([df_2017, df_iot])
    # Chuẩn hóa cột
    common_cols = list(set(df_train_raw.columns) - {'Label', 'Label_Bin', 'Label_Multi'})
    common_cols.sort()
    for c in common_cols:
        if c not in df_train_raw.columns: df_train_raw[c] = 0
    df_train_raw = df_train_raw[common_cols + ['Label_Bin', 'Label_Multi']]
    df_train_raw = clean_data(df_train_raw)
    
    print_dataset_stats(df_train_raw, "TOTAL TRAIN RAW (Merged)") # <--- Info tổng

    # ---------------------------------------------------------
    # BƯỚC 2: TRÍCH XUẤT DỮ LIỆU HIẾM (RESERVE) CHO STREAM
    # ---------------------------------------------------------
    print("\n[PHASE 2] TRÍCH XUẤT DỮ LIỆU HIẾM (RESERVE FOR STREAM)")
    rare_labels = [3, 6]
    reserved_data = []
    
    # Copy dữ liệu train để không ảnh hưởng
    df_train_processing = df_train_raw.copy()
    
    for lbl in rare_labels:
        rare_df = df_train_processing[df_train_processing['Label_Multi'] == lbl]
        if len(rare_df) > 0:
            sample = rare_df.sample(n=min(len(rare_df), 1000), replace=True, random_state=42)
            reserved_data.append(sample)
            print(f"   -> Đã dành riêng {len(sample)} mẫu Label {lbl} cho Stream.")
        else:
            print(f"   ⚠️ Cảnh báo: Không tìm thấy Label {lbl} trong tập Train gốc!")

    # ---------------------------------------------------------
    # BƯỚC 3: CÂN BẰNG TẬP TRAIN (SMOTE)
    # ---------------------------------------------------------
    print("\n[PHASE 3] CÂN BẰNG TẬP TRAIN (SMOTE + UNDERSAMPLING)")
    df_train_final = smart_smote_balance(df_train_processing, target_max=300000, target_min=50000)
    print_dataset_stats(df_train_final, "TRAIN FINAL (After SMOTE)") # <--- Info sau khi cân bằng
    
    # ---------------------------------------------------------
    # BƯỚC 4: TẠO TẬP STREAM & TRỘN DỮ LIỆU HIẾM
    # ---------------------------------------------------------
    print("\n[PHASE 4] TẠO TẬP STREAM ĐA DẠNG")
    s1 = read_dataset_optimized("CIC2018", 50000)
    s2 = read_dataset_optimized("Darknet", 50000)
    s3 = read_dataset_optimized("DoH", 50000)
    s4 = read_dataset_optimized("DDoS2019", 50000)
    
    stream_parts = [p for p in [s1, s2, s3, s4] if not p.empty]
    
    if reserved_data:
        stream_parts.extend(reserved_data)
        
    df_stream_raw = pd.concat(stream_parts)
    # Chuẩn hóa cột stream giống train
    for c in common_cols:
        if c not in df_stream_raw.columns: df_stream_raw[c] = 0
    df_stream_raw = df_stream_raw[common_cols + ['Label_Bin', 'Label_Multi']]
    df_stream_raw = clean_data(df_stream_raw)
    
    print_dataset_stats(df_stream_raw, "TOTAL STREAM RAW") # <--- Info tổng stream

    # ---------------------------------------------------------
    # BƯỚC 5: SCALING & LƯU FILE
    # ---------------------------------------------------------
    print("\n[PHASE 5] SCALING & SAVING")
    
    # Train scaler trên tập Train
    X_train = df_train_final[common_cols].values
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    # Lưu Scaler
    joblib.dump(scaler, os.path.join(OUTPUT_PATH, "train_scaler.pkl"))
    
    # Lưu tập Train
    df_train_save = pd.DataFrame(X_train_scaled, columns=common_cols)
    df_train_save['Label'] = df_train_final['Label_Bin'].values 
    df_train_save['Label_Multi'] = df_train_final['Label_Multi'].values
    df_train_save['Label_Bin'] = df_train_final['Label_Bin'].values
    
    df_train_save = shuffle(df_train_save, random_state=42).reset_index(drop=True)
    df_train_save.to_parquet(os.path.join(OUTPUT_PATH, "processed_initial_train_balanced.parquet"))
    
    # Transform tập Stream
    X_stream = df_stream_raw[common_cols].values
    X_stream_scaled = scaler.transform(X_stream)
    
    df_stream_save = pd.DataFrame(X_stream_scaled, columns=common_cols)
    df_stream_save['Label'] = df_stream_raw['Label_Bin'].values
    df_stream_save['Label_Multi'] = df_stream_raw['Label_Multi'].values
    df_stream_save['Label_Bin'] = df_stream_raw['Label_Bin'].values
    
    df_stream_save = shuffle(df_stream_save, random_state=42).reset_index(drop=True)
    df_stream_save.to_parquet(os.path.join(OUTPUT_PATH, "processed_online_stream.parquet"))
    
    # ---------------------------------------------------------
    # KẾT QUẢ
    # ---------------------------------------------------------
    analyze_quality(df_train_save, "FINAL TRAIN DATASET")
    analyze_quality(df_stream_save, "FINAL STREAM DATASET (INJECTED)")
    
    print(f"\n✅ HOÀN TẤT! Dữ liệu đã lưu tại: {OUTPUT_PATH}")

if __name__ == "__main__":
    main()