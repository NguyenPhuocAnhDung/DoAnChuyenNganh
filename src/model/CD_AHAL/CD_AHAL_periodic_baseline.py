import os
import time
import datetime
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, backend as K
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
from sklearn.utils import class_weight
from collections import deque
from tabulate import tabulate

# ==========================================
# 1. CẤU HÌNH CHO BASELINE "PERIODIC + RANDOM"
# ==========================================
# Mục tiêu: So sánh CD-AHAL với chiến lược Retrain định kỳ dùng Random Sampling (Không dùng Active Learning)
DATA_PATH = "../../../dataset/processed_v3_unified"
BASE_OUTPUT = "../../../baocao/PERIODIC_BASELINE_REVIEW3" # Folder riêng cho Baseline này
MODEL_PATH = os.path.join(BASE_OUTPUT, "models")
REPORT_PATH = os.path.join(BASE_OUTPUT, "reports_stream")
PLOT_PATH = os.path.join(BASE_OUTPUT, "plots_stream")
SOURCE_MODEL_PATH = "../../../baocao/CD_AHAL_FINAL_FULL_METRICS_FINAL/models" # Lấy model gốc để bắt đầu

for p in [REPORT_PATH, PLOT_PATH]: 
    if not os.path.exists(p): os.makedirs(p)

ONLINE_STREAM_FILE = os.path.join(DATA_PATH, "processed_online_stream.parquet")

TIME_STEPS = 10
BATCH_SIZE = 512
RAW_BATCH_SIZE = BATCH_SIZE * TIME_STEPS
REHEARSAL_SIZE = 5000
CLASS_NAMES = ['Normal', 'DoS', 'PortScan', 'Botnet', 'BruteForce', 'WebAttack', 'Infiltration', 'DDoS']

# --- CONFIG RIÊNG CHO PERIODIC BASELINE ---
RETRAIN_INTERVAL = 20  # Cứ mỗi 20 batch sẽ retrain 1 lần (Bất kể có drift hay không)
BUDGET_RATIO = 0.20    # Giống CD-AHAL (20% budget) nhưng lấy Random thay vì Oracle Entropy

# --- CUSTOM LAYERS (Bắt buộc để load model) ---
@tf.keras.utils.register_keras_serializable()
class AttentionLayer(layers.Layer):
    def __init__(self, **kwargs): super().__init__(**kwargs)
    def build(self, input_shape):
        self.W = self.add_weight(name='w', shape=(input_shape[-1], 1), initializer='normal')
        self.b = self.add_weight(name='b', shape=(input_shape[1], 1), initializer='zeros')
        super().build(input_shape)
    def call(self, x):
        e = K.tanh(K.dot(x, self.W) + self.b); a = K.softmax(e, axis=1)
        return K.sum(x * a, axis=1)

@tf.keras.utils.register_keras_serializable()
class MCDropout(layers.Dropout):
    def call(self, inputs, training=None): return super().call(inputs, training=True)

@tf.keras.utils.register_keras_serializable()
class F1Score(tf.keras.metrics.Metric):
    def __init__(self, name='f1_score', **kwargs):
        super(F1Score, self).__init__(name=name, **kwargs)
        self.precision = tf.keras.metrics.Precision(); self.recall = tf.keras.metrics.Recall()
    def update_state(self, y_true, y_pred, sample_weight=None):
        self.precision.update_state(y_true, y_pred, sample_weight)
        self.recall.update_state(y_true, y_pred, sample_weight)
    def result(self):
        p = self.precision.result(); r = self.recall.result()
        return 2 * ((p * r) / (p + r + tf.keras.backend.epsilon()))
    def reset_state(self): self.precision.reset_state(); self.recall.reset_state()

@tf.keras.utils.register_keras_serializable()
class SparseFocalLoss(tf.keras.losses.Loss):
    def __init__(self, gamma=2.0, alpha=0.25, **kwargs):
        super().__init__(**kwargs); self.gamma = gamma; self.alpha = alpha
        self.ce = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)
    def call(self, y_true, y_pred):
        ce = self.ce(y_true, y_pred); pt = tf.exp(-ce)
        return self.alpha * ((1 - pt) ** self.gamma) * ce
    def get_config(self): return {"gamma": self.gamma, "alpha": self.alpha}

# ==========================================
# HELPER
# ==========================================
class SmartBuffer:
    def __init__(self, max_size):
        self.max_size = max_size; self.buffer = [] 
    def add(self, x, y):
        for i in range(len(x)): self.buffer.append((x[i], y[i]))
        if len(self.buffer) > self.max_size:
            self.buffer = random.sample(self.buffer, self.max_size)
    def get_sample(self, n):
        if not self.buffer: return None, None
        s = random.sample(self.buffer, min(len(self.buffer), n))
        return np.array([x[0] for x in s]), np.array([x[1] for x in s])

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

def plot_visuals(batches, accs, retrain_indices, name):
    plt.figure(figsize=(14, 6))
    plt.plot(batches, accs, label='Periodic Accuracy', color='purple')
    if retrain_indices:
        vals = [accs[b] for b in retrain_indices]
        plt.scatter(retrain_indices, vals, color='black', zorder=5, label='Periodic Retrain', marker='D', s=40)
    plt.ylabel("Accuracy"); plt.legend(); plt.grid(True, alpha=0.3)
    plt.title(f"Baseline: Periodic Retraining (Every {RETRAIN_INTERVAL} batches) with Random Sampling")
    plt.savefig(os.path.join(PLOT_PATH, f"{name}.png"), dpi=300); plt.close()

# ==========================================
# MAIN PERIODIC BASELINE
# ==========================================
def main():
    print(f"--- BASELINE: PERIODIC RETRAINING (Every {RETRAIN_INTERVAL} Batches) ---")
    custom = {'AttentionLayer': AttentionLayer, 'MCDropout': MCDropout, 'F1Score': F1Score, 'SparseFocalLoss': SparseFocalLoss}
    
    # Load model gốc (chưa train phase 2) để công bằng
    try:
        model_A = keras.models.load_model(os.path.join(SOURCE_MODEL_PATH, "CD_AHAL_model_A_initial.h5"), custom_objects=custom)
        
        # [FIX] Re-compile để khởi tạo lại training loop cho môi trường hiện tại
        model_A.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        
        print("✅ Initial Models loaded successfully!")
    except:
        print(f"❌ Error loading models from {SOURCE_MODEL_PATH}. Check path!"); return

    df_stream = pd.read_parquet(ONLINE_STREAM_FILE)
    feat_cols = [c for c in df_stream.columns if c not in ['Label', 'Label_Multi', 'Label_Bin']]
    X_raw = df_stream[feat_cols].values; yb_raw = df_stream['Label'].values
    
    buffer = SmartBuffer(REHEARSAL_SIZE)
    log = {'batch': [], 'acc': [], 'retrain': []}
    
    n_batches = len(X_raw) // RAW_BATCH_SIZE
    all_true, all_pred = [], []
    
    print(f"-> Streaming {n_batches} batches with Periodic Retraining...")
    
    for i in range(n_batches):
        s = i * RAW_BATCH_SIZE; e = (i+1) * RAW_BATCH_SIZE
        X_seq, y_cur = prepare_sequences(X_raw[s:e], yb_raw[s:e], TIME_STEPS)
        if len(X_seq) == 0: continue
        
        # 1. Inference (Dùng model hiện tại)
        preds = (model_A.predict(X_seq, verbose=0) > 0.5).astype(int).flatten()
        acc = accuracy_score(y_cur, preds)
        
        all_true.extend(y_cur); all_pred.extend(preds)
        log['batch'].append(i); log['acc'].append(acc)
        
        # 2. Periodic Retraining Check
        # Reviewer 3: "So sánh với chiến lược cập nhật định kỳ"
        if (i + 1) % RETRAIN_INTERVAL == 0:
            log['retrain'].append(i)
            print(f"🔄 Periodic Retrain @ Batch {i} | Acc: {acc:.3f}")
            
            # A. Random Sampling (Thay vì Oracle Entropy)
            # Reviewer 3 muốn thấy hiệu quả của việc "chọn mẫu thông minh" (Active Learning)
            # nên baseline này phải chọn "ngẫu nhiên" để so sánh.
            budget_size = int(len(X_seq) * BUDGET_RATIO)
            
            # Chọn ngẫu nhiên index
            idx_random = np.random.choice(len(X_seq), budget_size, replace=False)
            X_train_new = X_seq[idx_random]
            y_train_new = y_cur[idx_random] # Giả định gán nhãn đúng cho mẫu ngẫu nhiên
            
            # B. Rehearsal Buffer
            X_buf, y_buf = buffer.get_sample(BATCH_SIZE)
            if X_buf is not None:
                X_train_new = np.concatenate([X_train_new, X_buf], axis=0)
                y_train_new = np.concatenate([y_train_new, y_buf], axis=0)
            
            # C. Update Model
            if len(X_train_new) > 0:
                cw = class_weight.compute_class_weight('balanced', classes=np.unique(y_train_new), y=y_train_new)
                cw_d = dict(enumerate(cw))
                model_A.fit(X_train_new, y_train_new, epochs=3, batch_size=64, verbose=0, class_weight=cw_d)
            
            # Add to buffer
            buffer.add(X_seq[idx_random], y_cur[idx_random])
        
        if i % 20 == 0: print(f"Batch {i}/{n_batches} | Acc: {acc:.3f}")

    # Summary
    print("\n--- PERIODIC BASELINE SUMMARY ---")
    plot_visuals(log['batch'], log['acc'], log['retrain'], "Periodic_Retrain_Performance")
    
    m_bin = calc_all_metrics(all_true, all_pred)
    df_res = pd.DataFrame([m_bin], index=['Periodic Baseline'])
    print(tabulate(df_res, headers='keys', tablefmt='grid'))
    
    df_res.to_csv(os.path.join(REPORT_PATH, "Periodic_Baseline_Metrics.csv"))
    print(f"✅ Done. Results saved to {REPORT_PATH}")

if __name__ == "__main__":
    main()