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
from sklearn.utils import class_weight
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix, classification_report

# ==========================================
# 1. CẤU HÌNH
# ==========================================
DATA_PATH = "D:/DoAnChuyenNganh/dataset/processed_v3_unified"
BASE_OUTPUT = "../../../baocao/CNN-Attention-BASELINE" # Folder riêng cho Baseline
MODEL_PATH = os.path.join(BASE_OUTPUT, "models")
PLOT_PATH = os.path.join(BASE_OUTPUT, "plots_train")
REPORT_PATH = os.path.join(BASE_OUTPUT, "reports_train")

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

def calc_metrics(y_true, y_pred, avg='weighted'):
    return {
        'Accuracy': accuracy_score(y_true, y_pred),
        'Precision': precision_score(y_true, y_pred, average=avg, zero_division=0),
        'Recall': recall_score(y_true, y_pred, average=avg, zero_division=0),
        'F1-Score': f1_score(y_true, y_pred, average=avg, zero_division=0)
    }

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
    
    # [BASELINE DIFFERENCE] Thay vì Attention, dùng GlobalPooling hoặc Flatten
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
    if not os.path.exists(INITIAL_TRAIN_FILE): print("❌ Missing Data"); return
    
    df = pd.read_parquet(INITIAL_TRAIN_FILE)
    feat_cols = [c for c in df.columns if c not in ['Label', 'Label_Multi', 'Label_Bin']]
    
    df_tr, df_tmp = train_test_split(df, test_size=0.3, stratify=df['Label'])
    df_val, df_test = train_test_split(df_tmp, test_size=0.5, stratify=df_tmp['Label'])
    
    def get_d(d):
        X, yb = prepare_sequences(d[feat_cols].values, d['Label'].values, TIME_STEPS)
        _, ym = prepare_sequences(d[feat_cols].values, d['Label_Multi'].values, TIME_STEPS)
        return X, yb, ym

    X_train, yb_train, ym_train = get_d(df_tr)
    X_val, yb_val, ym_val = get_d(df_val)
    X_test, yb_test, ym_test = get_d(df_test)
    
    # 1. Train Binary
    print("\nTraining Baseline Binary...")
    model_A = create_baseline_binary(X_train.shape[1:])
    hist_A = model_A.fit(X_train, yb_train, validation_data=(X_val, yb_val), 
                         epochs=EPOCHS, batch_size=BATCH_SIZE, 
                         callbacks=[EarlyStopping(patience=5, restore_best_weights=True)], verbose=1)
    model_A.save(os.path.join(MODEL_PATH, "Baseline_Binary.h5"))
    plot_history(hist_A, "History_Baseline_Binary.png")
    
    # 2. Train Multi
    print("\nTraining Baseline Multi...")
    idx = np.where(ym_train > 0)[0]; idx_v = np.where(ym_val > 0)[0]
    model_B = create_baseline_multi(X_train.shape[1:])
    hist_B = model_B.fit(X_train[idx], ym_train[idx], validation_data=(X_val[idx_v], ym_val[idx_v]),
                         epochs=EPOCHS, batch_size=BATCH_SIZE, 
                         callbacks=[EarlyStopping(patience=5)], verbose=1)
    model_B.save(os.path.join(MODEL_PATH, "Baseline_Multi.h5"))
    plot_history(hist_B, "History_Baseline_Multi.png")
    
    # 3. Eval
    print("\n--- BASELINE TEST EVALUATION ---")
    pb = (model_A.predict(X_test, verbose=0) > 0.5).astype(int).flatten()
    print("[Binary]"); print(calc_metrics(yb_test, pb))
    plot_cm(yb_test, pb, ['Normal', 'Attack'], "Baseline Binary CM", "Baseline_CM_Binary.png")
    
    pm = np.zeros_like(ym_test)
    idx = np.where(pb == 1)[0]
    if len(idx) > 0: pm[idx] = np.argmax(model_B.predict(X_test[idx], verbose=0), axis=1)
    print("[Multi]"); print(calc_metrics(ym_test, pm, 'weighted'))
    plot_cm(ym_test, pm, CLASS_NAMES, "Baseline Multi CM", "Baseline_CM_Multi.png", "Reds")
    
    print(f"✅ Baseline saved to {MODEL_PATH}")

if __name__ == "__main__":
    main()