import pandas as pd
import numpy as np
import os
import glob
import joblib
from sklearn.utils import shuffle
from sklearn.preprocessing import StandardScaler
from scipy.stats.mstats import winsorize

# ==========================================
# 1. CẤU HÌNH (MAX DATA CHO 32GB RAM)
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

OUTPUT_PATH = "../../dataset/processedstreamvs2.4"
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
# 2. MAPPING & CLEANING (ĐÃ FIX CHO FILE 'all.csv' CỦA BẠN)
# ==========================================
COLUMN_MAP_DICT = {
    # --- CICIDS2017 / 2018 / DDoS2019 ---
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
    'ack_flag_number': 'ACK Flag Count', 'urg_flag_number': 'URG Flag Count',
    'ece_flag_number': 'ECE Flag Count', 'cwe_flag_count': 'CWE Flag Count',
    'max': 'Max Packet Length', 'min': 'Min Packet Length', 
    'mean': 'Packet Length Mean', 'std': 'Packet Length Std',
    
    # --- [FIX MỚI] DARKNET (CIRA-CIC-DoHBrw-2020) MAPPING ---
    'Duration': 'Flow Duration',
    'FlowBytesSent': 'Total Length of Fwd Packets',
    'FlowBytesReceived': 'Total Length of Bwd Packets',
    'FlowSentRate': 'Fwd Packets/s',
    'FlowReceivedRate': 'Bwd Packets/s',
    'PacketLengthMean': 'Packet Length Mean',
    'PacketLengthStandardDeviation': 'Packet Length Std',
    'PacketLengthVariance': 'Packet Length Variance',
    'DoH': 'Label', # Quan trọng: Cột DoH chính là Label
    
    # --- CÁC BIẾN THỂ KHÁC ---
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
        
        # Ưu tiên map chính xác từ Dict (Case sensitive cho 'DoH' và 'Duration')
        if c_clean in COLUMN_MAP_DICT:
            final_name = COLUMN_MAP_DICT[c_clean]
        elif c_lower in COLUMN_MAP_DICT:
            final_name = COLUMN_MAP_DICT[c_lower]
        else:
            # Fallback
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
    
    # --- [FIX] XỬ LÝ NHÃN DoH/Darknet ---
    if lbl == 'true' or lbl == 'doh': return 7  # Attack (DoH)
    if lbl == 'false' or lbl == 'nondoh': return 0 # Normal
    
    # --- CÁC NHÃN KHÁC ---
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
    df.drop_duplicates(inplace=True)
    return df

def handle_outliers(df):
    cols = ['Flow Duration', 'Total Fwd Packets', 'Total Bwd Packets', 'Flow Packets/s']
    for col in cols:
        if col in df.columns:
            try: df[col] = winsorize(df[col], limits=[0.01, 0.01])
            except: pass
    return df

def print_stats(df, name="Dataset"):
    print(f"\n📊 THỐNG KÊ: {name}")
    print(f"   - Tổng mẫu: {len(df)}")
    if 'Label_Bin' in df.columns:
        c = df['Label_Bin'].value_counts()
        print(f"   - Normal (0): {c.get(0, 0)} | Attack (1): {c.get(1, 0)}")
    print("-" * 30)

def read_dataset_optimized(key, limit_benign=50000, limit_attack=50000):
    path_info = DIRS.get(key)
    if not path_info: return pd.DataFrame()
    path, ext, recursive = path_info
    
    if os.path.isfile(path): files = [path]
    else:
        if recursive: files = glob.glob(os.path.join(path, "**", f"*.{ext}"), recursive=True)
        else: files = glob.glob(os.path.join(path, f"*.{ext}"))

    print(f"--> Đang xử lý: {key} (Tìm thấy {len(files)} files)")
    
    df_list = []
    # Tăng max_read cho IoT2023/Darknet vì file nhỏ
    max_read = 200 if key in ["IoT2023", "Darknet"] else 50 
    count_ok = 0
    
    for i, f in enumerate(files):
        if count_ok >= 20 or i > max_read: break 
        
        try:
            if f.endswith('.parquet'): temp = pd.read_parquet(f)
            else: temp = pd.read_csv(f, encoding='latin1', on_bad_lines='skip', low_memory=False)
            
            if 'SourceIP' in temp.columns or 'Source IP' in temp.columns: 
                # [FIX] Darknet có SourceIP nhưng vẫn cần lấy, chỉ bỏ khi nó là IP rác hoàn toàn
                pass 

            temp = normalize_columns(temp)
            
            # [FIX] Darknet có ít cột, hạ min_req xuống 2
            cols_present = [c for c in TARGET_COLS if c in temp.columns]
            min_req = 2 
            if len(cols_present) < min_req: continue

            temp = temp[cols_present]
            if 'Label' not in temp.columns: continue
            
            temp['Label_Multi'] = temp['Label'].apply(standardize_label)
            temp['Label_Bin'] = temp['Label_Multi'].apply(lambda x: 0 if x==0 else 1)
            
            df_0 = temp[temp['Label_Bin'] == 0]
            df_1 = temp[temp['Label_Bin'] == 1]
            
            if len(df_0)>0: df_0 = df_0.sample(n=min(len(df_0), limit_benign))
            if len(df_1)>0: df_1 = df_1.sample(n=min(len(df_1), limit_attack))
            
            if not df_0.empty or not df_1.empty:
                df_list.append(pd.concat([df_0, df_1]))
                print(f"   + Đã đọc: {os.path.basename(f)} ({len(df_list[-1])} mẫu)")
                count_ok += 1
                
        except Exception as e: pass

    if not df_list: 
        print(f"    [!] CẢNH BÁO: {key} không có dữ liệu hợp lệ.")
        return pd.DataFrame()
        
    return pd.concat(df_list)

# ==========================================
# 3. MAIN (MAX DATA MODE)
# ==========================================
def main():
    print("--- BẮT ĐẦU PIPELINE (MAX DATA & DARKNET FINAL FIX) ---")
    
    # --- I. TRAIN (Tăng gấp đôi để Transformer hội tụ) ---
    print("\n[PHASE 1] INITIAL TRAIN (SCALING UP)")
    df_2017 = read_dataset_optimized("CIC2017", 800000, 800000) 
    df_iot  = read_dataset_optimized("IoT2023", 400000, 400000) 
    
    if df_2017.empty and df_iot.empty: return

    df_train = pd.concat([df_2017, df_iot])
    df_train = df_train.loc[:, ~df_train.columns.duplicated()]
    
    common_cols = list(set(df_train.columns) - {'Label', 'Label_Bin', 'Label_Multi'})
    common_cols.sort()
    
    for c in common_cols:
        if c not in df_train.columns: df_train[c] = 0
    df_train = df_train[common_cols + ['Label_Bin', 'Label_Multi']]

    print("   -> Cleaning & Scaling...")
    df_train = clean_data(df_train)
    df_train = handle_outliers(df_train)

    X_train = df_train[common_cols].values
    X_train = np.nan_to_num(X_train, nan=0.0, posinf=0.0, neginf=0.0)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    joblib.dump(scaler, os.path.join(OUTPUT_PATH, "train_scaler.pkl"))

    df_train_final = pd.DataFrame(X_train_scaled, columns=common_cols)
    df_train_final['Label'] = df_train['Label_Bin'].values
    df_train_final['Label_Multi'] = df_train['Label_Multi'].values
    df_train_final = shuffle(df_train_final, random_state=42).reset_index(drop=True)
    
    df_train_final.to_parquet(os.path.join(OUTPUT_PATH, "processed_initial_train_balanced.parquet"))
    print_stats(df_train_final, "INITIAL TRAIN")

    # --- II. STREAM ---
    print("\n[PHASE 2] ONLINE STREAM (LONG DURATION)")
    
    def get_stream_part(key, n_samples, label_val=None):
        df = read_dataset_optimized(key, n_samples*2, n_samples*2)
        if df.empty: return pd.DataFrame()
        for c in common_cols:
            if c not in df.columns: df[c] = 0
        df = df[common_cols + ['Label_Bin', 'Label_Multi']]
        if label_val is not None: df = df[df['Label_Bin'] == label_val]
        df = clean_data(df)
        if len(df) > n_samples: df = df.sample(n=n_samples, replace=False)
        return df

    print("   -> Tạo kịch bản Drift...")
    
    normal_filler = get_stream_part("CIC2018", 50000, label_val=0)
    
    # S1: Ổn định
    s1 = get_stream_part("CIC2018", 150000, label_val=0) 
    
    # S2: Darknet (Drift 1) - Giờ sẽ đọc được
    pure_dark = get_stream_part("Darknet", 100000) 
    filler_2 = normal_filler.sample(n=min(len(normal_filler), 20000), replace=True)
    s2 = pd.concat([pure_dark, filler_2])
    s2 = shuffle(s2).reset_index(drop=True)
    
    # S3: DoH (Drift 2)
    pure_doh = get_stream_part("DoH", 100000) 
    filler_3 = normal_filler.sample(n=min(len(normal_filler), 20000), replace=True)
    s3 = pd.concat([pure_doh, filler_3])
    s3 = shuffle(s3).reset_index(drop=True)
    
    # S4: DDoS19 (Zero-day)
    pure_ddos = get_stream_part("DDoS2019", 150000, label_val=1) 
    filler_4 = normal_filler.sample(n=min(len(normal_filler), 30000), replace=True)
    s4 = pd.concat([pure_ddos, filler_4])
    s4 = shuffle(s4).reset_index(drop=True)

    parts = [p for p in [s1, s2, s3, s4] if not p.empty]
    if not parts: print("LỖI: Stream Empty!"); return

    df_stream_raw = pd.concat(parts).reset_index(drop=True)
    
    print("   -> Scaling Stream...")
    X_stream = df_stream_raw[common_cols].values
    X_stream = np.nan_to_num(X_stream, nan=0.0, posinf=0.0, neginf=0.0)
    X_stream_scaled = scaler.transform(X_stream)
    
    df_stream_final = pd.DataFrame(X_stream_scaled, columns=common_cols)
    df_stream_final['Label'] = df_stream_raw['Label_Bin'].values
    df_stream_final['Label_Multi'] = df_stream_raw['Label_Multi'].values
    
    df_stream_final.to_parquet(os.path.join(OUTPUT_PATH, "processed_online_stream.parquet"))
    print_stats(df_stream_final, "ONLINE STREAM")

if __name__ == "__main__":
    main()