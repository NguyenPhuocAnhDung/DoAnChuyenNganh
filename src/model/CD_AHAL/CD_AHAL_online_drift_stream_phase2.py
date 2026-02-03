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
# 1. CẤU HÌNH & IMPORT
# ==========================================
DATA_PATH = "../../../dataset/processed_v3_unified"
BASE_OUTPUT = "../../../baocao/CD_AHAL_FINAL_FULL_METRICS_FINAL"
MODEL_PATH = os.path.join(BASE_OUTPUT, "models")
REPORT_PATH = os.path.join(BASE_OUTPUT, "reports_stream")
PLOT_PATH = os.path.join(BASE_OUTPUT, "plots_stream")
PHASE3_EXPORT_PATH = "../../../baocao/CD_AHAL_FINAL_FULL_METRICS"

for p in [REPORT_PATH, PLOT_PATH, PHASE3_EXPORT_PATH]: 
    if not os.path.exists(p): os.makedirs(p)

ONLINE_STREAM_FILE = os.path.join(DATA_PATH, "processed_online_stream.parquet")

TIME_STEPS = 10
BATCH_SIZE = 512
RAW_BATCH_SIZE = BATCH_SIZE * TIME_STEPS
MC_SAMPLES = 20 
REHEARSAL_SIZE = 5000
CLASS_NAMES = ['Normal', 'DoS', 'PortScan', 'Botnet', 'BruteForce', 'WebAttack', 'Infiltration', 'DDoS']

# --- AL CONFIG ---
ORACLE_BUDGET_RATIO = 0.20  
PSEUDO_CONF_THRESH = 0.95   

# --- CUSTOM LAYERS & METRICS ---
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
# 2. HELPER FUNCTIONS
# ==========================================
class SmartBuffer:
    def __init__(self, max_size):
        self.max_size = max_size; self.buffer = [] 
    def add(self, x, y):
        for i in range(len(x)): self.buffer.append((x[i], y[i]))
        if len(self.buffer) > self.max_size:
            self.buffer = random.sample(self.buffer, self.max_size)
    def get_sample(self, n):
        if not self.buffer: return None
        s = random.sample(self.buffer, min(len(self.buffer), n))
        return np.array([x[0] for x in s]), np.array([x[1] for x in s])

def prepare_sequences(X, y, time_steps):
    if len(X) < time_steps: return np.array([]), np.array([])
    n = len(X) // time_steps
    X_seq = X[:n*time_steps].reshape((n, time_steps, X.shape[1]))
    y_seq = y[time_steps-1::time_steps]
    return X_seq, y_seq

@tf.function
def get_mc_predictions(model, x, n_samples):
    return tf.stack([model(x, training=True) for _ in range(n_samples)], axis=0)

def calculate_entropy(probs):
    epsilon = 1e-10
    p = np.clip(probs, epsilon, 1 - epsilon)
    entropy = - (p * np.log(p) + (1 - p) * np.log(1 - p))
    return entropy

class DriftStats:
    def __init__(self): self.events = []
    def log(self, batch, latency, rec_acc, interval):
        self.events.append({"Batch": batch, "Time": datetime.datetime.now().strftime("%H:%M:%S"), 
                            "Latency(ms)": round(latency,1), "Recov_Acc": round(rec_acc,4), "Interval": interval})
    def save(self):
        if not self.events: return
        df = pd.DataFrame(self.events)
        df.to_csv(os.path.join(REPORT_PATH, "Drift_Log.csv"), index=False)
        print("\n>>> DRIFT EVENT LOG:")
        print(tabulate(df, headers='keys', tablefmt='grid'))

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

def plot_drift_visuals(batches, accs, losses, drift_indices):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True)
    ax1.plot(batches, accs, label='Real-time Accuracy', color='#1f77b4')
    if drift_indices:
        vals = [accs[b] for b in drift_indices]
        ax1.scatter(drift_indices, vals, color='red', zorder=5, label='Drift Detected', marker='x', s=80)
    ax1.set_ylabel("Accuracy"); ax1.legend(); ax1.grid(True, alpha=0.3)
    ax1.set_title("Phase 2: Stream Accuracy & Drift Events", fontweight='bold')
    
    ax2.plot(batches, losses, label='Real-time Loss', color='orange')
    ax2.set_ylabel("Loss"); ax2.set_xlabel("Batch Index"); ax2.legend(); ax2.grid(True, alpha=0.3)
    plt.tight_layout(); plt.savefig(os.path.join(PLOT_PATH, "Phase2_Drift_Analysis.png"), dpi=300); plt.close()

# ==========================================
# 3. MAIN STREAM (SAFE VERSION - NO 5x PENALTY)
# ==========================================
def main():
    print("--- PHASE 2: ONLINE STREAMING WITH ACTIVE LEARNING ---")
    custom = {'AttentionLayer': AttentionLayer, 'MCDropout': MCDropout, 'F1Score': F1Score, 'SparseFocalLoss': SparseFocalLoss}
    
    try:
        model_A = keras.models.load_model(os.path.join(MODEL_PATH, "CD_AHAL_model_A_initial.h5"), custom_objects=custom)
        model_B = keras.models.load_model(os.path.join(MODEL_PATH, "CD_AHAL_model_B_initial.h5"), custom_objects=custom)
        model_A.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        print("✅ Models loaded successfully!")
    except: print("❌ Error loading models."); return

    if not os.path.exists(ONLINE_STREAM_FILE): print("❌ Data file not found."); return
    df_stream = pd.read_parquet(ONLINE_STREAM_FILE)
    feat_cols = [c for c in df_stream.columns if c not in ['Label', 'Label_Multi', 'Label_Bin']]
    
    X_raw = df_stream[feat_cols].values
    yb_raw = df_stream['Label'].values; ym_raw = df_stream['Label_Multi'].values
    
    buffer = SmartBuffer(REHEARSAL_SIZE)
    entropy_hist = deque(maxlen=20)
    log = {'batch': [], 'acc': [], 'loss': [], 'drift': []}
    drift_stats = DriftStats()
    n_batches = len(X_raw) // RAW_BATCH_SIZE
    
    all_true_b, all_pred_b = [], []
    all_true_m, all_pred_m = [], []
    last_drift = -1

    print(f"-> Streaming {n_batches} batches...")
    for i in range(n_batches):
        s = i * RAW_BATCH_SIZE; e = (i+1) * RAW_BATCH_SIZE
        X_seq, yb_cur = prepare_sequences(X_raw[s:e], yb_raw[s:e], TIME_STEPS)
        _, ym_cur = prepare_sequences(X_raw[s:e], ym_raw[s:e], TIME_STEPS)
        if len(X_seq) == 0: continue
        
        # 1. Inference
        # Chạy mô hình MC_SAMPLES lần (20 lần)
        mc_preds = get_mc_predictions(model_A, X_seq, MC_SAMPLES).numpy()
        # Tính trung bình cộng kết quả dự đoán
        mean_preds = np.mean(mc_preds, axis=0).flatten()
        batch_entropy = calculate_entropy(mean_preds)
        avg_entropy = np.mean(batch_entropy)
        
        pb = (mean_preds > 0.5).astype(int)
        res = model_A.evaluate(X_seq, yb_cur, verbose=0)
        batch_loss, batch_acc = res[0], res[1]
        
        all_true_b.extend(yb_cur); all_pred_b.extend(pb)

        # 2. Drift Check
        is_drift = False
        if len(entropy_hist) >= 5:
            if avg_entropy > np.mean(entropy_hist) + 3 * np.std(entropy_hist) and avg_entropy > 0.1: is_drift = True
        if batch_acc < 0.90: is_drift = True 
        
        entropy_hist.append(avg_entropy)
        
        # 3. Adaptation
        if is_drift:
            t0 = time.time()
            log['drift'].append(i)
            
            # A. Oracle ORACLE_BUDGET_RATIO lấy 20% mẫu có entropy cao nhất
            budget_size = int(len(X_seq) * ORACLE_BUDGET_RATIO)
            idx_sorted_entropy = np.argsort(batch_entropy)[::-1] 
            idx_oracle = idx_sorted_entropy[:budget_size]
            X_oracle = X_seq[idx_oracle]; y_oracle = yb_cur[idx_oracle]
            
            # B. Pseudo
            mask_pseudo = (mean_preds > PSEUDO_CONF_THRESH) | (mean_preds < (1 - PSEUDO_CONF_THRESH))
            idx_pseudo_cand = np.where(mask_pseudo)[0]
            idx_pseudo = np.setdiff1d(idx_pseudo_cand, idx_oracle)
            X_pseudo = X_seq[idx_pseudo]
            y_pseudo = (mean_preds[idx_pseudo] > 0.5).astype(int)
            
            # C. Buffer
            if buffer.buffer:
                X_buf, y_buf = buffer.get_sample(BATCH_SIZE)
            else:
                X_buf = np.empty((0, TIME_STEPS, X_seq.shape[2]))
                y_buf = np.empty((0,))

            # D. Combine
            list_X = [X_oracle]
            list_y = [y_oracle] # Lấy nhãn thật từ dữ liệu
            if len(X_pseudo) > 0:
                list_X.append(X_pseudo)
                list_y.append(y_pseudo)
            if len(X_buf) > 0:
                list_X.append(X_buf)
                list_y.append(y_buf)
            
            X_train_upd = np.concatenate(list_X, axis=0)
            y_train_upd = np.concatenate(list_y, axis=0)

            # E. Update (BALANCED DEFAULT - NO 5x PENALTY)
            if len(X_train_upd) > 0:
                cw = class_weight.compute_class_weight('balanced', classes=np.unique(y_train_upd), y=y_train_upd)
                cw_d = dict(enumerate(cw))
                # Đã xóa dòng lệnh phạt cw_d[1] *= 5.0 để đảm bảo ổn định
                model_A.fit(X_train_upd, y_train_upd, epochs=3, batch_size=64, verbose=0, class_weight=cw_d)
            
            # Log
            new_acc = accuracy_score(yb_cur, (model_A.predict(X_seq, verbose=0)>0.5).astype(int))
            lat = (time.time() - t0) * 1000
            drift_stats.log(i, lat, new_acc, i - last_drift if last_drift!=-1 else 0)
            print(f"⚠️ Drift @ {i} | Ent: {avg_entropy:.3f} | Oracle: {len(X_oracle)} | Acc: {batch_acc:.2f}->{new_acc:.2f}")
            last_drift = i
            
            buffer.add(X_oracle, y_oracle)

        log['batch'].append(i); log['acc'].append(batch_acc); log['loss'].append(batch_loss)
        
        # Multiclass Prediction
        idx = np.where(pb == 1)[0]
        pm_cur = np.zeros_like(ym_cur)
        if len(idx) > 0: pm_cur[idx] = np.argmax(model_B.predict(X_seq[idx], verbose=0), axis=1)
        all_true_m.extend(ym_cur); all_pred_m.extend(pm_cur)

        if i % 20 == 0: print(f"Batch {i}/{n_batches} | Acc: {batch_acc:.3f}")

    # Reports
    print("\n--- PHASE 2 SUMMARY REPORT ---")
    drift_stats.save()
    plot_drift_visuals(log['batch'], log['acc'], log['loss'], log['drift'])
    
    m_bin = calc_all_metrics(all_true_b, all_pred_b)
    m_bin["Avg Loss"] = np.mean(log['loss'])
    m_mul = calc_all_metrics(all_true_m, all_pred_m, 'weighted')
    
    summary_df = pd.DataFrame([m_bin, m_mul], index=['Binary Phase 2', 'Multiclass Phase 2'])
    print(tabulate(summary_df, headers='keys', tablefmt='grid'))
    summary_df.to_csv(os.path.join(REPORT_PATH, "Overall_Phase2_Summary.csv"))
    
    plot_cm(all_true_b, all_pred_b, ['Normal', 'Attack'], "Stream_Binary_CM", "Blues")
    plot_cm(all_true_m, all_pred_m, CLASS_NAMES, "Stream_Multi_CM", "Reds")
    
    print(f"\n📦 EXPORTING MODELS FOR PHASE 3...")
    model_A.save(os.path.join(PHASE3_EXPORT_PATH, "CD_AHAL_Binary_Phase2_Final.h5"))
    model_B.save(os.path.join(PHASE3_EXPORT_PATH, "CD_AHAL_Multi_Phase2_Final.h5"))
    print(f"✅ All Results saved to {PLOT_PATH}")

if __name__ == "__main__":
    main()