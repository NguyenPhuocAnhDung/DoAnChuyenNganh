from sklearn import metrics
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model
from tensorflow.keras.callbacks import EarlyStopping
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns 
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.utils import class_weight 
from collections import deque
import random
import os
import time
import shap 
from tabulate import tabulate
import datetime

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# ==========================================
# 1. CẤU HÌNH HỆ THỐNG
# ==========================================
DATA_PATH = "../../dataset/processed/processedstreamvs2.4"
# [LƯU Ý] Kiểm tra lại đường dẫn này cho khớp với máy bạn (active hay ative)
PLOT_PATH = "../../baocao/main_active_learning/plots" 
MODEL_PATH = "../../baocao/main_active_learning/models"
REPORT_PATH = "../../baocao/main_active_learning/reports"

# Tạo thư mục nếu chưa có
for p in [PLOT_PATH, MODEL_PATH, REPORT_PATH]:
    if not os.path.exists(p): os.makedirs(p)

INITIAL_TRAIN_FILE = os.path.join(DATA_PATH, "processed_initial_train_balanced.parquet")
ONLINE_STREAM_FILE = os.path.join(DATA_PATH, "processed_online_stream.parquet")
VALIDATION_FILE = os.path.join(DATA_PATH, "processed_validation.parquet") 

# --- HYPERPARAMETERS ---
TIME_STEPS = 10 
N_CLASSES_MULTI = 8 
BATCH_SIZE = 256
RAW_BATCH_SIZE = BATCH_SIZE * TIME_STEPS 
DROPOUT_RATE = 0.3
MC_SAMPLES = 10 
REHEARSAL_SAMPLE_SIZE = 4096 
BUFFER_SIZE = 20000

# [CẤU HÌNH TRAINING]
INITIAL_EPOCHS = 30 
ONLINE_EPOCHS = 25 
ES_PATIENCE = 3 

# [CẤU HÌNH DRIFT]
UNCERTAINTY_WINDOW = 50 
NORMAL_CLASS_BOOST = 3.0 
DRIFT_THRESHOLD_BASE = 3.0

CLASS_NAMES = ['Normal', 'DoS', 'PortScan', 'Botnet', 'BruteForce', 'WebAttack', 'Infiltration', 'DDoS']
FEATURE_NAMES = [] 

# ==========================================
# 2. MODULE GHI LOG & VISUALIZATION
# ==========================================
class DriftRecorder:
    def __init__(self, filename="Drift_Events_Detailed.csv"):
        self.filepath = os.path.join(REPORT_PATH, filename)
        self.events = []
        self.columns = [
            "Event_ID", "Batch_Index", "Timestamp", 
            "Drift_Uncertainty", "Drift_Threshold",
            "Pre_Drift_Acc", "Drift_Point_Acc", "Post_Adapt_Acc",
            "Performance_Drop", "Recovery_Gain",
            "Adaptation_Latency_ms", "Drift_Status"
        ]

    def log_event(self, batch_idx, unc, thresh, pre_acc, drift_acc, post_acc, latency):
        event_id = len(self.events) + 1
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        drop = pre_acc - drift_acc
        gain = post_acc - drift_acc
        
        record = {
            "Event_ID": event_id,
            "Batch_Index": batch_idx,
            "Timestamp": timestamp,
            "Drift_Uncertainty": round(unc, 5),
            "Drift_Threshold": round(thresh, 5),
            "Pre_Drift_Acc": round(pre_acc, 4),
            "Drift_Point_Acc": round(drift_acc, 4),
            "Post_Adapt_Acc": round(post_acc, 4),
            "Performance_Drop": round(drop, 4),
            "Recovery_Gain": round(gain, 4),
            "Adaptation_Latency_ms": round(latency, 2),
            "Drift_Status": "Detected & Adapted"
        }
        self.events.append(record)

    def save_to_csv(self):
        if not self.events:
            print("ℹ️ Không có sự kiện Drift nào được ghi nhận.")
            return
        df = pd.DataFrame(self.events)
        df = df[self.columns]
        df.to_csv(self.filepath, index=False)
        print(f"\n✅ [REPORT] Đã xuất file báo cáo chi tiết tại: {self.filepath}")
        print("\n" + "="*80)
        print("📋 DRIFT EVENTS SUMMARY TABLE")
        print("="*80)
        print(tabulate(df[["Event_ID", "Batch_Index", "Drift_Point_Acc", "Post_Adapt_Acc", "Recovery_Gain"]], 
                       headers="keys", tablefmt="grid"))

def plot_training_history_5_panels(history, filename="Training_History_Full.png"):
    """Vẽ 5 biểu đồ (Loss, Acc, F1, Precision, Recall)"""
    metrics = ['loss', 'accuracy', 'f1_score', 'precision', 'recall']
    titles = ['Model Loss', 'Model Accuracy', 'Model F1_score', 'Model Precision', 'Model Recall']
    keys = list(history.history.keys())
    
    plt.figure(figsize=(18, 10))
    for i, metric_name in enumerate(metrics):
        train_key = next((k for k in keys if metric_name in k and 'val' not in k), None)
        val_key = next((k for k in keys if 'val_' in k and metric_name in k), None)
        
        if train_key:
            plt.subplot(2, 3, i + 1)
            plt.plot(history.history[train_key], label=f'Train {metric_name}')
            if val_key:
                plt.plot(history.history[val_key], label=f'Val {metric_name}')
            plt.title(titles[i])
            plt.ylabel(titles[i].split(' ')[1])
            plt.xlabel('Epoch')
            plt.legend(loc='best')
            plt.grid(True, alpha=0.3)
            
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_PATH, filename))
    plt.close()
    print(f"✅ [PLOT] Đã lưu biểu đồ Training History tại: {PLOT_PATH}/{filename}")

def plot_stream_results(metrics_df, drift_indices, plot_path):
    """Vẽ Figure 5 (Comparison) và Figure 6 (Uncertainty & Drift)"""
    batches = metrics_df['batch'].values
    
    # --- FIGURE 5: ACCURACY COMPARISON (WITH BASELINE EXPLICIT LEGEND) ---
    plt.figure(figsize=(14, 7))
    
    # Đường giải pháp đề xuất
    plt.plot(batches, metrics_df['acc_dynamic'], 
             label='Proposed Method (Active Learning + Drift Adapt)', 
             color='#1f77b4', linewidth=2)
    
    # Đường Baseline (Mô hình tĩnh)
    plt.plot(batches, metrics_df['acc_static'], 
             label='Baseline (Static Model - No Retraining)', 
             color='#d62728', linestyle='--', linewidth=2, alpha=0.8)
    
    plt.title("Figure 5: Performance Comparison - Proposed Adaptive Model vs. Static Baseline", fontsize=14, fontweight='bold')
    plt.xlabel("Stream Batches (Time)", fontsize=12)
    plt.ylabel("Accuracy", fontsize=12)
    plt.legend(fontsize=11, loc='lower left', frameon=True, shadow=True)
    plt.grid(True, alpha=0.3, linestyle='-.')
    plt.ylim(0.0, 1.05)
    
    # Annotation Gain
    if len(batches) > 10:
        last_dyn = metrics_df['acc_dynamic'].iloc[-1]
        last_stat = metrics_df['acc_static'].iloc[-1]
        gain = last_dyn - last_stat
        plt.annotate(f'Performance Gain: +{gain:.2%}', 
                     xy=(batches[-1], last_dyn), 
                     xytext=(batches[-1] - len(batches)*0.2, last_dyn - 0.1),
                     arrowprops=dict(facecolor='green', shrink=0.05),
                     fontsize=12, color='green', fontweight='bold')

    plt.savefig(os.path.join(plot_path, "Figure_5_Accuracy_Comparison_Explicit.png"), dpi=300)
    plt.close()
    print(f"✅ [PLOT] Đã xuất Figure 5 (Baseline Comparison) tại thư mục plots.")

    # --- FIGURE 6: UNCERTAINTY & DRIFT INTERVALS ---
    plt.figure(figsize=(14, 7))
    plt.plot(batches, metrics_df['unc'], label='Model Uncertainty Score', color='purple', linewidth=1.5)
    
    for i, idx in enumerate(drift_indices):
        label = 'Drift Event (Retraining Triggered)' if i == 0 else ""
        plt.axvline(x=idx, color='orange', linestyle='-', alpha=0.8)
        plt.axvspan(max(0, idx-2), min(max(batches), idx+2), color='orange', alpha=0.3, label=label)

    plt.title("Figure 6: Uncertainty Monitoring & Drift Detection Points", fontsize=14, fontweight='bold')
    plt.xlabel("Stream Batches", fontsize=12)
    plt.ylabel("Uncertainty (Entropy)", fontsize=12)
    plt.legend(fontsize=11, loc='upper left')
    plt.grid(True, alpha=0.3)
    
    plt.savefig(os.path.join(plot_path, "Figure_6_Uncertainty_Drift_Explicit.png"), dpi=300)
    plt.close()
    print(f"✅ [PLOT] Đã xuất Figure 6 (Uncertainty) tại thư mục plots.")

# ==========================================
# 3. CÁC CLASS LOGIC & UTILS (CUSTOM OBJECTS)
# ==========================================

# [FIX] ĐĂNG KÝ CLASS ĐỂ SERIALIZATION (SỬA LỖI CLONE_MODEL)
@tf.keras.utils.register_keras_serializable()
class F1Score(tf.keras.metrics.Metric):
    def __init__(self, name='f1_score', **kwargs):
        super(F1Score, self).__init__(name=name, **kwargs)
        self.precision = tf.keras.metrics.Precision()
        self.recall = tf.keras.metrics.Recall()
    def update_state(self, y_true, y_pred, sample_weight=None):
        self.precision.update_state(y_true, y_pred, sample_weight)
        self.recall.update_state(y_true, y_pred, sample_weight)
    def result(self):
        p = self.precision.result(); r = self.recall.result()
        return 2 * ((p * r) / (p + r + tf.keras.backend.epsilon()))
    def reset_state(self):
        self.precision.reset_state(); self.recall.reset_state()

@tf.keras.utils.register_keras_serializable()
class SparseFocalLoss(tf.keras.losses.Loss):
    def __init__(self, gamma=2.0, alpha=0.25, **kwargs):
        super().__init__(**kwargs)
        self.gamma = gamma; self.alpha = alpha
        self.ce = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)
    def call(self, y_true, y_pred):
        ce = self.ce(y_true, y_pred); pt = tf.exp(-ce)
        return self.alpha * ((1 - pt) ** self.gamma) * ce

# [FIX] ĐƯA MCDROPOUT RA NGOÀI VÀ ĐĂNG KÝ
@tf.keras.utils.register_keras_serializable()
class MCDropout(layers.Dropout):
    def call(self, inputs, training=None):
        return super().call(inputs, training=True)

class DynamicController:
    def __init__(self, base_lr=1e-5, base_thresh=3.0):
        self.base_lr = base_lr; self.base_thresh = base_thresh; self.ema_uncertainty = 0.01 
    def update(self, current_uncertainty):
        self.ema_uncertainty = 0.9 * self.ema_uncertainty + 0.1 * current_uncertainty
        ratio = current_uncertainty / (self.ema_uncertainty + 1e-9)
        new_lr = min(self.base_lr * max(1.0, ratio * 20.0), 0.001) 
        new_thresh = max(1.5, self.base_thresh / max(1.0, np.log1p(ratio)))
        return new_lr, new_thresh

class SmartRehearsalBuffer:
    def __init__(self, max_size):
        self.max_size = max_size; self.buffer = [] 
    def add(self, x, y, unc):
        for i in range(len(x)): self.buffer.append((x[i], y[i], unc))
        self.buffer.sort(key=lambda x: x[2], reverse=True)
        if len(self.buffer) > self.max_size:
            n_h = int(self.max_size * 0.6); n_r = self.max_size - n_h
            self.buffer = self.buffer[:n_h] + random.sample(self.buffer[n_h:], n_r)
    def get_sample(self, n):
        if not self.buffer: return np.array([]), np.array([])
        s = random.sample(self.buffer, min(len(self.buffer), n))
        return np.array([x[0] for x in s]), np.array([x[1] for x in s])
    def get_all_X(self, limit=50):
        if not self.buffer: return np.array([])
        return np.array([x[0] for x in random.sample(self.buffer, min(len(self.buffer), limit))])

class MulticlassRehearsalBuffer:
    def __init__(self, max_s, n_c):
        self.bufs = {i: deque(maxlen=max_s//(n_c-1)) for i in range(1, n_c)}
    def add(self, x, y):
        for i in range(len(x)):
            if y[i] in self.bufs: self.bufs[y[i]].append((x[i], y[i]))
    def get_sample(self, n):
        res = []; acts = [k for k in self.bufs if self.bufs[k]]
        if not acts: return np.array([]), np.array([])
        pc = max(1, n//len(acts))
        for k in acts: res.extend(random.sample(self.bufs[k], min(len(self.bufs[k]), pc)))
        random.shuffle(res); batch = list(zip(*res))
        return np.array(batch[0]), np.array(batch[1])

# ==========================================
# 4. MODEL CREATION
# ==========================================
def prepare_sequences(X, y, time_steps):
    if len(X) < time_steps: return np.array([]), np.array([])
    n = len(X) // time_steps
    X_seq = X[:n*time_steps].reshape((n, time_steps, X.shape[1]))
    y_seq = y[time_steps-1::time_steps]
    return X_seq, y_seq

def create_binary_model(input_shape):
    inputs = layers.Input(shape=input_shape)
    
    # [FIX] Sử dụng MCDropout đã đăng ký ở global
    x = layers.Conv1D(128, 3, activation='relu', padding='same')(inputs)
    x = layers.BatchNormalization()(x); x = layers.MaxPooling1D(2)(x); x = MCDropout(DROPOUT_RATE)(x)
    x = layers.Conv1D(64, 3, activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x); x = MCDropout(DROPOUT_RATE)(x)
    x = layers.Bidirectional(layers.GRU(128, return_sequences=True))(x); x = MCDropout(DROPOUT_RATE)(x)
    x = layers.Bidirectional(layers.GRU(64, return_sequences=False))(x)
    outputs = layers.Dense(1, activation='sigmoid')(x)
    model = Model(inputs, outputs)
    
    # 4 metrics quan trọng cho đồ thị history
    metrics = [
        'accuracy', 
        F1Score(),
        tf.keras.metrics.Precision(name='precision'),
        tf.keras.metrics.Recall(name='recall')
    ]
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=metrics)
    return model

def create_multiclass_model(input_shape):
    inputs = layers.Input(shape=input_shape)
    # [FIX] Sử dụng MCDropout đã đăng ký ở global
    x = layers.Conv1D(64, 3, activation='relu', padding='same')(inputs)
    x = layers.BatchNormalization()(x); x = layers.MaxPooling1D(2)(x); x = MCDropout(DROPOUT_RATE)(x)
    x = layers.Bidirectional(layers.GRU(64, return_sequences=False))(x); x = MCDropout(DROPOUT_RATE)(x)
    outputs = layers.Dense(N_CLASSES_MULTI, activation='softmax')(x)
    model = Model(inputs, outputs)
    model.compile(optimizer='adam', loss=SparseFocalLoss(), metrics=['accuracy'])
    return model

@tf.function(reduce_retracing=True)
def get_uncertainty(model, x):
    preds = tf.stack([model(x, training=False) for _ in range(MC_SAMPLES)], axis=0)
    return tf.reduce_mean(tf.math.reduce_variance(preds, axis=0))

# ==========================================
# 5. XAI & UTILS
# ==========================================
def explain_drift_shap(model, background_data, drift_data, batch_idx):
    try:
        bg_2d = np.mean(background_data, axis=1)
        drift_2d = np.mean(drift_data[:20], axis=1) 
        def model_predict_wrapper(x_2d):
            x_3d = np.expand_dims(x_2d, axis=1).repeat(TIME_STEPS, axis=1)
            return model.predict(x_3d, verbose=0)
        explainer = shap.KernelExplainer(model_predict_wrapper, bg_2d[:30])
        with np.errstate(divide='ignore', invalid='ignore'):
            shap_values = explainer.shap_values(drift_2d, nsamples=100, silent=True)
        if isinstance(shap_values, list): sv = shap_values[0]
        else: sv = shap_values
        
        plt.figure(figsize=(10, 6))
        shap.summary_plot(sv, drift_2d, feature_names=FEATURE_NAMES, show=False)
        plt.title(f"Drift Explanation - Batch {batch_idx}")
        plt.tight_layout()
        plt.savefig(os.path.join(PLOT_PATH, f"XAI_Batch_{batch_idx}.png"))
        plt.close()
    except Exception: pass

# ==========================================
# 6. MAIN EXECUTION
# ==========================================
def main():
    print(">>> STARTING CNN-GRU SYSTEM (WITH FIGURE 5 & 6 GENERATION) <<<")
    
    # 1. LOAD & PREPARE DATA
    df_init = pd.read_parquet(INITIAL_TRAIN_FILE)
    feat_cols = [c for c in df_init.columns if c not in ['Label', 'Label_Multi']]
    global FEATURE_NAMES; FEATURE_NAMES = feat_cols
    X_init = df_init[feat_cols].values; y_init_bin = df_init['Label'].values; y_init_multi = df_init['Label_Multi'].values
    X_seq, y_bin_seq = prepare_sequences(X_init, y_init_bin, TIME_STEPS)
    _, y_multi_seq = prepare_sequences(X_init, y_init_multi, TIME_STEPS)

    # 2. INITIAL TRAIN (DYNAMIC MODEL)
    early = EarlyStopping(monitor='val_loss', patience=ES_PATIENCE, restore_best_weights=True)
    print("\n--- Phase 1: Binary Training (Initial) ---")
    model_A = create_binary_model((TIME_STEPS, len(feat_cols)))
    cw_A = class_weight.compute_class_weight('balanced', classes=np.unique(y_bin_seq), y=y_bin_seq)
    
    # Lưu history để vẽ biểu đồ 5 panels
    history_A = model_A.fit(X_seq, y_bin_seq, epochs=INITIAL_EPOCHS, batch_size=256, validation_split=0.1, 
                            callbacks=[early], class_weight=dict(enumerate(cw_A)), verbose=1)
    
    # Vẽ và lưu biểu đồ History
    plot_training_history_5_panels(history_A, filename="Training_History_Full.png")
    
    buffer_A = SmartRehearsalBuffer(BUFFER_SIZE); buffer_A.add(X_seq, y_bin_seq, 0.0)

    # [QUAN TRỌNG] TẠO STATIC BASELINE MODEL (Để vẽ Figure 5)
    print("\n--- Creating Static Baseline Model for Comparison ---")
    
    # [FIXED] Bây giờ clone_model sẽ hoạt động vì class đã được đăng ký
    model_static = keras.models.clone_model(model_A)
    model_static.set_weights(model_A.get_weights()) # Copy trọng số ban đầu
    model_static.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    
    # Multiclass Init
    model_B = create_multiclass_model((TIME_STEPS, len(feat_cols)))
    idx_atk = np.where(y_multi_seq > 0)[0]
    buffer_B = MulticlassRehearsalBuffer(BUFFER_SIZE, N_CLASSES_MULTI)
    if len(idx_atk) > 0:
        X_B, y_B = X_seq[idx_atk], y_multi_seq[idx_atk]; buffer_B.add(X_B, y_B)
        cw_B = class_weight.compute_class_weight('balanced', classes=np.unique(y_B), y=y_B)
        model_B.fit(X_B, y_B, epochs=INITIAL_EPOCHS, batch_size=256, validation_split=0.1, verbose=0)

    # 3. STREAMING & LOGGING
    print("\n>>> START ONLINE STREAMING <<<")
    df_stream = pd.read_parquet(ONLINE_STREAM_FILE)
    X_st_raw = df_stream[feat_cols].values; y_st_bin = df_stream['Label'].values; y_st_multi = df_stream['Label_Multi'].values
    
    unc_hist = deque(maxlen=UNCERTAINTY_WINDOW)
    recorder = DriftRecorder(filename="Drift_Events_Detailed.csv")
    controller = DynamicController(base_lr=1e-5, base_thresh=3.0)
    opt_A = keras.optimizers.Adam(learning_rate=1e-5); opt_B = keras.optimizers.Adam(learning_rate=1e-5)
    
    # [ADD] Thêm 'acc_static' vào đây
    stream_history = {
        'batch': [],
        'acc_dynamic': [],
        'acc_static': [],   # <--- BỔ SUNG
        'unc': [],
        'f1': [],      
        'latency': []   
    }
    drift_indices = []

    n_batches = len(X_st_raw) // RAW_BATCH_SIZE
    
    for i in range(n_batches):
        start = i * RAW_BATCH_SIZE; end = (i+1) * RAW_BATCH_SIZE
        X_curr = X_st_raw[start:end]
        X_sq, y_bin = prepare_sequences(X_curr, y_st_bin[start:end], TIME_STEPS)
        _, y_mul = prepare_sequences(X_curr, y_st_multi[start:end], TIME_STEPS)
        if len(X_sq) == 0: continue
        
        # --- A. DYNAMIC MODEL PREDICT ---
        # [FIX 2] Đo latency cho từng batch
        t_start = time.time()
        p_A = model_A.predict(X_sq, verbose=0)
        batch_latency = (time.time() - t_start) * 1000 # ms
        
        pred_bin = (p_A > 0.5).astype(int).flatten()
        acc_dyn = accuracy_score(y_bin, pred_bin)
        
        # [FIX 3] Tính F1 Score
        from sklearn.metrics import f1_score
        f1_dyn = f1_score(y_bin, pred_bin, zero_division=0)
        
        unc = get_uncertainty(model_A, X_sq).numpy()

        # --- B. STATIC MODEL PREDICT (BASELINE) ---
        p_static = model_static.predict(X_sq, verbose=0)
        pred_static_bin = (p_static > 0.5).astype(int).flatten()
        acc_stat = accuracy_score(y_bin, pred_static_bin)

        # --- C. LOGGING DATA ---
        stream_history['batch'].append(i)
        stream_history['acc_dynamic'].append(acc_dyn)
        stream_history['acc_static'].append(acc_stat) # <--- LOG THÊM
        stream_history['unc'].append(unc)
        stream_history['f1'].append(f1_dyn)      
        stream_history['latency'].append(batch_latency) 

        # Multiclass Prediction (Dynamic)
        idx_p = np.where(pred_bin == 1)[0]
        if len(idx_p) > 0:
            p_B = model_B.predict(X_sq[idx_p], verbose=0)

        # --- D. DRIFT DETECTION & ADAPTATION ---
        dyn_lr, dyn_thresh = controller.update(unc)
        opt_A.learning_rate.assign(dyn_lr); opt_B.learning_rate.assign(dyn_lr)
        
        is_drift = False
        if len(unc_hist) == UNCERTAINTY_WINDOW:
            if unc > np.mean(unc_hist) + dyn_thresh * np.std(unc_hist) and unc > 0.001: 
                is_drift = True
        unc_hist.append(unc)
        
        if is_drift:
            print(f"[DRIFT DETECTED] Batch {i} | Unc: {unc:.4f} | Acc: {acc_dyn:.4f}")
            drift_indices.append(i) # Lưu lại vị trí để vẽ Figure 6
            
            # Tính Pre-Acc
            pre_acc = np.mean(stream_history['acc_dynamic'][-6:-1]) if len(stream_history['acc_dynamic']) > 5 else acc_dyn
            t0 = time.time()
            
            # XAI
            bg = buffer_A.get_all_X(limit=20)
            if len(bg)>0: explain_drift_shap(model_A, bg, X_sq, i)
            
            # Retrain Model A (Active Learning)
            X_old, y_old = buffer_A.get_sample(REHEARSAL_SAMPLE_SIZE)
            if len(X_old) > 0:
                X_ft = np.concatenate((X_sq, X_old)); y_ft = np.concatenate((y_bin, y_old))
                cw = class_weight.compute_class_weight('balanced', classes=np.unique(y_ft), y=y_ft)
                cw_d = dict(enumerate(cw))
                if 0 in cw_d: cw_d[0] *= NORMAL_CLASS_BOOST
                model_A.compile(optimizer=opt_A, loss='binary_crossentropy', metrics=['accuracy'])
                model_A.fit(X_ft, y_ft, epochs=ONLINE_EPOCHS, verbose=0, class_weight=cw_d)
            
            # Retrain Model B
            idx_t = np.where(y_mul > 0)[0]
            if len(idx_t) > 0:
                X_old_B, y_old_B = buffer_B.get_sample(REHEARSAL_SAMPLE_SIZE)
                X_ft_B = np.concatenate((X_sq[idx_t], X_old_B)); y_ft_B = np.concatenate((y_mul[idx_t], y_old_B))
                model_B.compile(optimizer=opt_B, loss=SparseFocalLoss(), metrics=['accuracy'])
                model_B.fit(X_ft_B, y_ft_B, epochs=ONLINE_EPOCHS, verbose=0)
            
            latency_adapt = (time.time() - t0) * 1000
            
            # Đánh giá phục hồi
            post_pred = (model_A.predict(X_sq, verbose=0) > 0.5).astype(int).flatten()
            post_acc = accuracy_score(y_bin, post_pred)
            
            # Ghi vào CSV Report
            recorder.log_event(i, unc, dyn_thresh, pre_acc, acc_dyn, post_acc, latency_adapt)
            unc_hist.clear()
            
        buffer_A.add(X_sq, y_bin, unc)
        if np.any(y_mul > 0): buffer_B.add(X_sq[y_mul > 0], y_mul[y_mul > 0])
        
        if i % 20 == 0: print(f"Batch {i}/{n_batches} processed. Acc Dyn: {acc_dyn:.3f} | Acc Stat: {acc_stat:.3f}")

    # 4. FINAL EXPORT
    print("\nProcessing Final Results & Plots...")
    
    # Xuất CSV
    recorder.save_to_csv()
    
    # Xuất Figure 5 & 6
    metrics_df = pd.DataFrame(stream_history)
    plot_stream_results(metrics_df, drift_indices, PLOT_PATH)
    
    # Lưu Models
    model_A.save(os.path.join(MODEL_PATH, "final_model_A.h5"))
    model_B.save(os.path.join(MODEL_PATH, "final_model_B.h5"))
    
    # Validation Độc lập (Nếu có)
    if os.path.exists(VALIDATION_FILE):
        print("\n--- Running Final Validation ---")
        df_val = pd.read_parquet(VALIDATION_FILE)
        X_val_seq, y_val_seq = prepare_sequences(df_val[feat_cols].values, df_val['Label'].values, TIME_STEPS)
        pred_val = (model_A.predict(X_val_seq, verbose=0) > 0.5).astype(int).flatten()
        print(classification_report(y_val_seq, pred_val, target_names=['Normal', 'Attack']))
        
        plt.figure(figsize=(6, 5))
        sns.heatmap(confusion_matrix(y_val_seq, pred_val), annot=True, fmt='d', cmap='Greens')
        plt.title("Validation Confusion Matrix")
        plt.savefig(os.path.join(PLOT_PATH, "CM_Validation_Final.png"))
        plt.close()

    # [FIX 4] SỬA LỖI XUẤT FILE COMPARE
    # Bỏ check 'if metrics in locals' vì biến đó không tồn tại
    print("\n--- Saving History for Comparison ---")
    df_compare = pd.DataFrame({
        'batch': stream_history['batch'],
        'accuracy': stream_history['acc_dynamic'],
        'accuracy_static': stream_history['acc_static'], # <--- BỔ SUNG
        'f1': stream_history['f1'],
        'latency': stream_history['latency']
    })
    # Đảm bảo thư mục tồn tại
    if not os.path.exists(PLOT_PATH): os.makedirs(PLOT_PATH)
    
    save_path_cmp = os.path.join(PLOT_PATH, "history_active_learning.csv")
    df_compare.to_csv(save_path_cmp, index=False)
    print(f"✅ [COMPARE] Đã lưu lịch sử Active Learning tại: {save_path_cmp}")

    # in các dòng drift...
    print("DONE.")

if __name__ == "__main__":
    main()