import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, backend as K
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix, classification_report
from tabulate import tabulate

# ==========================================
# 1. CẤU HÌNH
# ==========================================
DATA_PATH = "D:/DoAnChuyenNganh/dataset/processed_v3_unified"
BASE_OUTPUT = "../../../baocao/GRU_ATTENTION_BASELINE"
MODEL_PATH = os.path.join(BASE_OUTPUT, "models")
REPORT_PATH = os.path.join(BASE_OUTPUT, "reports_stream")
PLOT_PATH = os.path.join(BASE_OUTPUT, "plots_stream")

for p in [REPORT_PATH, PLOT_PATH]: 
    if not os.path.exists(p): os.makedirs(p)

ONLINE_STREAM_FILE = os.path.join(DATA_PATH, "processed_online_stream.parquet")

TIME_STEPS = 10
BATCH_SIZE = 512
RAW_BATCH_SIZE = BATCH_SIZE * TIME_STEPS
CLASS_NAMES = ['Normal', 'DoS', 'PortScan', 'Botnet', 'BruteForce', 'WebAttack', 'Infiltration', 'DDoS']

# --- CUSTOM LAYERS (Bắt buộc để load model) ---
@tf.keras.utils.register_keras_serializable()
class AttentionLayer(layers.Layer):
    def __init__(self, **kwargs): super(AttentionLayer, self).__init__(**kwargs)
    def build(self, input_shape):
        self.W = self.add_weight(name='w', shape=(input_shape[-1], 1), initializer='normal')
        self.b = self.add_weight(name='b', shape=(input_shape[1], 1), initializer='zeros')
        super().build(input_shape)
    def call(self, x):
        e = K.tanh(K.dot(x, self.W) + self.b); a = K.softmax(e, axis=1)
        return K.sum(x * a, axis=1)

# ==========================================
# 2. UTILS
# ==========================================
def prepare_sequences(X, y, time_steps):
    if len(X) < time_steps: return np.array([]), np.array([])
    n = len(X) // time_steps
    X_seq = X[:n*time_steps].reshape((n, time_steps, X.shape[1]))
    y_seq = y[time_steps-1::time_steps]
    return X_seq, y_seq

def calc_all_metrics(y_true, y_pred, avg='weighted'):
    return {
        'Accuracy': accuracy_score(y_true, y_pred),
        'Precision': precision_score(y_true, y_pred, average=avg, zero_division=0),
        'Recall': recall_score(y_true, y_pred, average=avg, zero_division=0),
        'F1-Score': f1_score(y_true, y_pred, average=avg, zero_division=0)
    }

def plot_cm(y_true, y_pred, classes, name, cmap='Blues'):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap=cmap, xticklabels=classes, yticklabels=classes)
    plt.title(name); plt.ylabel('True'); plt.xlabel('Predicted')
    plt.savefig(os.path.join(PLOT_PATH, f"{name}.png"), bbox_inches='tight'); plt.close()

# ==========================================
# 3. MAIN STREAM (STATIC)
# ==========================================
def main():
    print("--- BASELINE PHASE 2: STATIC STREAMING (GRU-ATTENTION) ---")
    custom = {'AttentionLayer': AttentionLayer}
    
    try:
        model_A = keras.models.load_model(os.path.join(MODEL_PATH, "Baseline_GRU_Att_Binary.h5"), custom_objects=custom)
        model_B = keras.models.load_model(os.path.join(MODEL_PATH, "Baseline_GRU_Att_Multi.h5"), custom_objects=custom)
        print("✅ Baseline Models loaded!")
    except Exception as e: print(f"❌ Error: {e}"); return

    df_stream = pd.read_parquet(ONLINE_STREAM_FILE)
    feat_cols = [c for c in df_stream.columns if c not in ['Label', 'Label_Multi', 'Label_Bin']]
    X_raw = df_stream[feat_cols].values
    yb_raw = df_stream['Label'].values; ym_raw = df_stream['Label_Multi'].values
    
    log_acc = []
    log_batch = []
    all_true_b, all_pred_b = [], []
    all_true_m, all_pred_m = [], []

    n_batches = len(X_raw) // RAW_BATCH_SIZE
    print(f"-> Streaming {n_batches} batches...")
    
    for i in range(n_batches):
        s = i * RAW_BATCH_SIZE; e = (i+1) * RAW_BATCH_SIZE
        X_seq, yb_cur = prepare_sequences(X_raw[s:e], yb_raw[s:e], TIME_STEPS)
        _, ym_cur = prepare_sequences(X_raw[s:e], ym_raw[s:e], TIME_STEPS)
        if len(X_seq) == 0: continue
        
        # Predict (Static)
        pb = (model_A.predict(X_seq, verbose=0) > 0.5).astype(int).flatten()
        all_true_b.extend(yb_cur); all_pred_b.extend(pb)
        
        acc = accuracy_score(yb_cur, pb)
        log_batch.append(i); log_acc.append(acc)
        
        # Multi
        idx = np.where(pb == 1)[0]
        pm_cur = np.zeros_like(ym_cur)
        if len(idx) > 0: pm_cur[idx] = np.argmax(model_B.predict(X_seq[idx], verbose=0), axis=1)
        all_true_m.extend(ym_cur); all_pred_m.extend(pm_cur)

        if i % 20 == 0: print(f"Batch {i}/{n_batches} | Static Acc: {acc:.3f}")

    # Final Report
    print("\n--- BASELINE STREAM REPORT ---")
    
    # Save Metrics for Comparison
    df_log = pd.DataFrame({'Batch': log_batch, 'Baseline_Acc': log_acc})
    df_log.to_csv(os.path.join(REPORT_PATH, "Baseline_GRU_Att_Stream_Metrics.csv"), index=False)
    
    plt.figure(figsize=(14, 6))
    plt.plot(log_batch, log_acc, label='Baseline (Static) Accuracy', color='grey', linestyle='--')
    plt.title("Baseline GRU-Attention Performance (Static)"); plt.legend(); plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(PLOT_PATH, "Baseline_Stream_Acc.png"), dpi=300); plt.close()
    
    # Summary
    m_bin = calc_all_metrics(all_true_b, all_pred_b)
    m_mul = calc_all_metrics(all_true_m, all_pred_m, 'weighted')
    summary_df = pd.DataFrame([m_bin, m_mul], index=['Binary', 'Multiclass'])
    print("\n>>> BASELINE SUMMARY:")
    print(tabulate(summary_df, headers='keys', tablefmt='grid'))
    summary_df.to_csv(os.path.join(REPORT_PATH, "Baseline_Overall_Summary.csv"))
    
    plot_cm(all_true_b, all_pred_b, ['Normal', 'Attack'], "Stream_Binary_CM")
    plot_cm(all_true_m, all_pred_m, CLASS_NAMES, "Stream_Multi_CM", "Reds")
    
    # Export
    model_A.save(os.path.join(MODEL_PATH, "Baseline_GRU_Att_Binary_Phase2.h5"))
    model_B.save(os.path.join(MODEL_PATH, "Baseline_GRU_Att_Multi_Phase2.h5"))
    
    print(f"✅ Baseline Complete. Saved to {MODEL_PATH}")

if __name__ == "__main__":
    main()