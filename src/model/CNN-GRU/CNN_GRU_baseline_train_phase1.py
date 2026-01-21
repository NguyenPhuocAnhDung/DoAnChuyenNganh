import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix, classification_report

# ==========================================
# 1. CẤU HÌNH
# ==========================================
DATA_PATH = "D:/DoAnChuyenNganh/dataset/processed_v3_unified"
BASE_OUTPUT = "../../../baocao/CNN-GRU_BASELINE" # Folder riêng cho Baseline
MODEL_PATH = os.path.join(BASE_OUTPUT, "models")
PLOT_PATH = os.path.join(BASE_OUTPUT, "plots_train")
REPORT_PATH = os.path.join(BASE_OUTPUT, "reports_train")

# Tạo thư mục nếu chưa tồn tại
for p in [MODEL_PATH, PLOT_PATH, REPORT_PATH]: 
    if not os.path.exists(p): os.makedirs(p)

INITIAL_TRAIN_FILE = os.path.join(DATA_PATH, "processed_initial_train_balanced.parquet")

TIME_STEPS = 10
N_CLASSES_MULTI = 8
DROPOUT_RATE = 0.3
EPOCHS = 100
BATCH_SIZE = 512
CLASS_NAMES = ['Normal', 'DoS', 'PortScan', 'Botnet', 'BruteForce', 'WebAttack', 'Infiltration', 'DDoS']

# ==========================================
# 2. UTILS
# ==========================================
def prepare_sequences(X, y, time_steps):
    if len(X) < time_steps: return np.array([]), np.array([])
    n = len(X) // time_steps
    X_seq = X[:n*time_steps].reshape((n, time_steps, X.shape[1]))
    y_seq = y[time_steps-1::time_steps]
    return X_seq, y_seq

def plot_history(history, filename):
    metrics = ['loss', 'accuracy']
    plt.figure(figsize=(12, 5))
    for i, m in enumerate(metrics):
        if m in history.history:
            plt.subplot(1, 2, i+1)
            plt.plot(history.history[m], label=f'Train {m}')
            plt.plot(history.history[f'val_{m}'], label=f'Val {m}')
            plt.title(m.upper()); plt.legend(); plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(PLOT_PATH, filename), dpi=300); plt.close()

def plot_cm(y_true, y_pred, classes, title, filename, cmap='Blues'):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap=cmap, xticklabels=classes, yticklabels=classes)
    plt.title(title); plt.ylabel('True'); plt.xlabel('Predicted')
    plt.savefig(os.path.join(PLOT_PATH, filename), bbox_inches='tight', dpi=300); plt.close()

def save_classification_report(y_true, y_pred, target_names, filename_csv):
    """
    Tính toán Precision, Recall, F1, Accuracy chi tiết và lưu file CSV
    """
    # 1. In ra console
    print(f"\n--- Report: {filename_csv} ---")
    report_str = classification_report(y_true, y_pred, target_names=target_names, digits=4, zero_division=0)
    print(report_str)
    
    # 2. Lưu vào file CSV
    report_dict = classification_report(y_true, y_pred, target_names=target_names, output_dict=True, digits=4, zero_division=0)
    df_report = pd.DataFrame(report_dict).transpose()
    df_report.to_csv(os.path.join(REPORT_PATH, filename_csv))
    print(f"📄 Saved report to {os.path.join(REPORT_PATH, filename_csv)}")

# ==========================================
# 3. BASELINE MODEL (CNN-GRU, NO ATTENTION)
# ==========================================
def create_baseline_binary(input_shape):
    inputs = layers.Input(shape=input_shape)
    # CNN Layer
    x = layers.Conv1D(128, 3, activation='relu', padding='same')(inputs)
    x = layers.MaxPooling1D(2)(x)
    x = layers.Dropout(DROPOUT_RATE)(x)
    
    # Bi-GRU Layer
    x = layers.Bidirectional(layers.GRU(128, return_sequences=True))(x)
    
    # [BASELINE DIFFERENCE] Thay vì Attention, dùng GlobalMaxPooling
    x = layers.GlobalMaxPooling1D()(x) 
    
    x = layers.Dropout(DROPOUT_RATE)(x)
    outputs = layers.Dense(1, activation='sigmoid')(x)
    
    model = Model(inputs, outputs, name="Baseline_Binary")
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

def create_baseline_multi(input_shape):
    inputs = layers.Input(shape=input_shape)
    x = layers.Conv1D(64, 3, activation='relu', padding='same')(inputs)
    x = layers.MaxPooling1D(2)(x)
    x = layers.Dropout(DROPOUT_RATE)(x)
    
    x = layers.Bidirectional(layers.GRU(64, return_sequences=True))(x)
    
    # [BASELINE DIFFERENCE] No Attention
    x = layers.GlobalMaxPooling1D()(x)
    
    x = layers.Dropout(DROPOUT_RATE)(x)
    outputs = layers.Dense(N_CLASSES_MULTI, activation='softmax')(x)
    
    model = Model(inputs, outputs, name="Baseline_Multi")
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

# ==========================================
# 4. MAIN
# ==========================================
def main():
    print("--- BASELINE PHASE 1: TRAIN CNN-GRU (NO ATTENTION) ---")
    if not os.path.exists(INITIAL_TRAIN_FILE): 
        print(f"❌ Missing Data at {INITIAL_TRAIN_FILE}")
        return
    
    df = pd.read_parquet(INITIAL_TRAIN_FILE)
    # Loại bỏ các cột label khỏi features
    feat_cols = [c for c in df.columns if c not in ['Label', 'Label_Multi', 'Label_Bin']]
    
    # Chia dữ liệu: Train 70%, Val 15%, Test 15%
    df_tr, df_tmp = train_test_split(df, test_size=0.3, stratify=df['Label'])
    df_val, df_test = train_test_split(df_tmp, test_size=0.5, stratify=df_tmp['Label'])
    
    def get_d(d):
        X, yb = prepare_sequences(d[feat_cols].values, d['Label'].values, TIME_STEPS)
        _, ym = prepare_sequences(d[feat_cols].values, d['Label_Multi'].values, TIME_STEPS)
        return X, yb, ym

    X_train, yb_train, ym_train = get_d(df_tr)
    X_val, yb_val, ym_val = get_d(df_val)
    X_test, yb_test, ym_test = get_d(df_test)
    
    # --------------------------------------
    # 1. Train Binary Model
    # --------------------------------------
    print("\nTraining Baseline Binary...")
    model_A = create_baseline_binary(X_train.shape[1:])
    hist_A = model_A.fit(X_train, yb_train, validation_data=(X_val, yb_val), 
                         epochs=EPOCHS, batch_size=BATCH_SIZE, 
                         callbacks=[EarlyStopping(patience=5, restore_best_weights=True)], verbose=1)
    model_A.save(os.path.join(MODEL_PATH, "Baseline_Binary.h5"))
    plot_history(hist_A, "History_Baseline_Binary.png")
    
    # --------------------------------------
    # 2. Train Multi Model (Only on Attack samples)
    # --------------------------------------
    print("\nTraining Baseline Multi...")
    idx = np.where(ym_train > 0)[0]    # Chỉ lấy mẫu tấn công
    idx_v = np.where(ym_val > 0)[0]    # Chỉ lấy mẫu tấn công
    
    model_B = create_baseline_multi(X_train.shape[1:])
    hist_B = model_B.fit(X_train[idx], ym_train[idx], validation_data=(X_val[idx_v], ym_val[idx_v]),
                         epochs=EPOCHS, batch_size=BATCH_SIZE, 
                         callbacks=[EarlyStopping(patience=5)], verbose=1)
    model_B.save(os.path.join(MODEL_PATH, "Baseline_Multi.h5"))
    plot_history(hist_B, "History_Baseline_Multi.png")
    
    # --------------------------------------
    # 3. EVALUATION (Full Metrics)
    # --------------------------------------
    print("\n--- BASELINE TEST EVALUATION ---")
    
    # --- Đánh giá Binary ---
    print("\n[Evaluation] Binary Classification (Normal vs Attack)")
    pb_prob = model_A.predict(X_test, verbose=0)
    pb = (pb_prob > 0.5).astype(int).flatten()
    
    plot_cm(yb_test, pb, ['Normal', 'Attack'], "Baseline Binary CM", "Baseline_CM_Binary.png")
    save_classification_report(yb_test, pb, ['Normal', 'Attack'], "Report_Binary.csv")
    
    # --- Đánh giá Multi-class (Hierarchical) ---
    print("\n[Evaluation] Multi-class Classification")
    
    # Logic: Mặc định gán là Normal (0)
    pm = np.zeros_like(ym_test)
    
    # Tìm các mẫu mà model Binary dự đoán là Attack (1)
    idx_attack_pred = np.where(pb == 1)[0]
    
    if len(idx_attack_pred) > 0:
        # Đưa các mẫu này vào model Multi để phân loại cụ thể
        pm_probs = model_B.predict(X_test[idx_attack_pred], verbose=0)
        # Gán nhãn cụ thể (argmax trả về 0-7, nhưng trong ngữ cảnh model B train với label gốc nên map đúng)
        pm[idx_attack_pred] = np.argmax(pm_probs, axis=1)
    
    plot_cm(ym_test, pm, CLASS_NAMES, "Baseline Multi CM", "Baseline_CM_Multi.png", "Reds")
    save_classification_report(ym_test, pm, CLASS_NAMES, "Report_Multi.csv")
    
    print(f"✅ Baseline evaluation complete. Models saved to {MODEL_PATH}")

if __name__ == "__main__":
    main()