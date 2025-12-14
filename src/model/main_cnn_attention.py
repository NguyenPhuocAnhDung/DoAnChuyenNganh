import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model
from tensorflow.keras.callbacks import EarlyStopping
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, precision_recall_fscore_support
from sklearn.utils import class_weight
from collections import deque
import random
import os
import time
import shap

# Tắt log cảnh báo
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# ==========================================
# 1. CẤU HÌNH HỆ THỐNG
# ==========================================
DATA_PATH = "../../dataset/processed/processedstreamvs2.4"
PLOT_PATH = "../../baocao/main_cnn_attention/plots"
MODEL_PATH = "../../baocao/main_cnn_attention/models"
REPORT_PATH = "../../baocao/main_cnn_attention/reports"

if not os.path.exists(PLOT_PATH): os.makedirs(PLOT_PATH)
if not os.path.exists(MODEL_PATH): os.makedirs(MODEL_PATH)
if not os.path.exists(REPORT_PATH): os.makedirs(REPORT_PATH)

INITIAL_TRAIN_FILE = os.path.join(DATA_PATH, "processed_initial_train_balanced.parquet")
ONLINE_STREAM_FILE = os.path.join(DATA_PATH, "processed_online_stream.parquet")

# --- HYPERPARAMETERS ---
TIME_STEPS = 10
N_CLASSES_MULTI = 8
BATCH_SIZE = 256
RAW_BATCH_SIZE = BATCH_SIZE * TIME_STEPS
DROPOUT_RATE = 0.3
MC_SAMPLES = 10
REHEARSAL_SAMPLE_SIZE = 4096
BUFFER_SIZE = 20000

INITIAL_EPOCHS = 25
ONLINE_EPOCHS = 25
ES_PATIENCE = 3

UNCERTAINTY_WINDOW = 50
NORMAL_CLASS_BOOST = 3.0

CLASS_NAMES = ['Normal', 'DoS', 'PortScan', 'Botnet', 'BruteForce', 'WebAttack', 'Infiltration', 'DDoS']
FEATURE_NAMES = []

# ==========================================
# 2. MODULES & UTIL CLASSES
# ==========================================
class DynamicController:
    def __init__(self, base_lr=1e-5, base_thresh=3.0):
        self.base_lr = base_lr; self.base_thresh = base_thresh; self.ema_uncertainty = 0.01
    def update(self, current_uncertainty):
        self.ema_uncertainty = 0.9 * self.ema_uncertainty + 0.1 * current_uncertainty
        ratio = current_uncertainty / (self.ema_uncertainty + 1e-9)
        new_lr = min(self.base_lr * max(1.0, ratio * 20.0), 0.001)
        new_thresh = max(1.5, self.base_thresh / max(1.0, np.log1p(ratio)))
        return new_lr, new_thresh

def generate_admin_report(shap_values, feature_names, drift_data):
    feature_importance = np.mean(np.abs(shap_values), axis=0)
    top_indices = np.argsort(feature_importance)[::-1][:3]
    print("\n" + "!"*50)
    print("🚨 [ADMIN REPORT] PHÂN TÍCH NGUYÊN NHÂN TẤN CÔNG 🚨")
    print("!"*50)
    for idx in top_indices:
        idx = int(idx)
        feat_name = feature_names[idx] if idx < len(feature_names) else f"Feature_{idx}"
        score = float(feature_importance[idx])
        mean_val = float(np.mean(drift_data[:, idx]))
        print(f"👉 Tác nhân chính: {feat_name} (Score: {score:.4f}, Avg: {mean_val:.4f})")
    print("-" * 40)

def explain_drift_shap(model, background_data, drift_data, batch_idx):
    print(f"\n   🔍 [XAI] Đang chạy SHAP Batch {batch_idx}...")
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
        generate_admin_report(sv, FEATURE_NAMES, drift_2d)
        
        plt.figure(figsize=(10, 6))
        shap.summary_plot(sv, drift_2d, feature_names=FEATURE_NAMES, show=False)
        plt.title(f"Drift Explanation - Batch {batch_idx}")
        plt.tight_layout()
        plt.savefig(os.path.join(PLOT_PATH, f"XAI_Batch_{batch_idx}.png"))
        plt.close()
    except Exception as e: print(f"   ⚠️ [XAI Warning]: {e}")

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

class SparseFocalLoss(tf.keras.losses.Loss):
    def __init__(self, gamma=2.0, alpha=0.25):
        super().__init__()
        self.gamma = gamma; self.alpha = alpha
        self.ce = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)
    def call(self, y_true, y_pred):
        ce = self.ce(y_true, y_pred); pt = tf.exp(-ce)
        return self.alpha * ((1 - pt) ** self.gamma) * ce

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

def prepare_sequences(X, y, time_steps):
    if len(X) < time_steps: return np.array([]), np.array([])
    n = len(X) // time_steps
    X_seq = X[:n*time_steps].reshape((n, time_steps, X.shape[1]))
    y_seq = y[time_steps-1::time_steps]
    return X_seq, y_seq

# ==============================================================================
# MODEL DEFINITIONS
# ==============================================================================
def create_binary_model(input_shape):
    inputs = layers.Input(shape=input_shape)
    class MCDropout(layers.Dropout):
        def call(self, i): return super().call(i, training=True)
    
    x = layers.Conv1D(128, 3, activation='relu', padding='same')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling1D(2)(x)
    x = MCDropout(DROPOUT_RATE)(x)
    
    x = layers.Conv1D(64, 3, activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = MCDropout(DROPOUT_RATE)(x)
    
    attention = layers.Attention(name="cnn_self_attention")([x, x])
    x = layers.GlobalAveragePooling1D()(attention)
    
    outputs = layers.Dense(1, activation='sigmoid')(x)
    model = Model(inputs, outputs, name="Pure_CNN_Attention_Binary")
    
    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy',
                           tf.keras.metrics.Precision(name='precision'),
                           tf.keras.metrics.Recall(name='recall'),
                           F1Score()])
    return model

def create_multiclass_model(input_shape):
    inputs = layers.Input(shape=input_shape)
    class MCDropout(layers.Dropout):
        def call(self, i): return super().call(i, training=True)
    
    x = layers.Conv1D(64, 3, activation='relu', padding='same')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling1D(2)(x)
    x = MCDropout(DROPOUT_RATE)(x)
    
    attention = layers.Attention(name="cnn_self_attention_multi")([x, x])
    x = layers.GlobalAveragePooling1D()(attention)
    
    outputs = layers.Dense(N_CLASSES_MULTI, activation='softmax')(x)
    model = Model(inputs, outputs, name="Pure_CNN_Attention_Multiclass")
    model.compile(optimizer='adam', loss=SparseFocalLoss(), metrics=['accuracy'])
    return model

@tf.function(reduce_retracing=True)
def get_uncertainty(model, x):
    preds = tf.stack([model(x, training=False) for _ in range(MC_SAMPLES)], axis=0)
    return tf.reduce_mean(tf.math.reduce_variance(preds, axis=0))

# ==========================================
# 3. HELPER BÁO CÁO & EVALUATION
# ==========================================

# --- HÀM VẼ LEARNING CURVES (Chuẩn Format 5 hình) ---
def plot_training_history(history, phase_name):
    """
    Vẽ đồ thị Loss, Accuracy, F1, Precision, Recall cho quá trình huấn luyện
    """
    hist = history.history
    epochs = range(1, len(hist['loss']) + 1)
    
    # Xác định các metrics có sẵn
    has_f1 = 'f1_score' in hist
    has_prec = 'precision' in hist
    has_rec = 'recall' in hist
    
    # Tạo subplot: 2 hàng, 3 cột (để đủ chỗ cho 5 hình)
    rows = 2
    cols = 3
    fig, axes = plt.subplots(rows, cols, figsize=(18, 10))
    axes = axes.flatten()
    
    # 1. Plot Loss
    axes[0].plot(epochs, hist['loss'], 'b-o', label='Train Loss')
    if 'val_loss' in hist:
        axes[0].plot(epochs, hist['val_loss'], 'r-o', label='Val Loss')
    axes[0].set_title(f'{phase_name} - Loss')
    axes[0].legend()
    axes[0].grid(True)
    
    # 2. Plot Accuracy
    axes[1].plot(epochs, hist['accuracy'], 'b-o', label='Train Acc')
    if 'val_accuracy' in hist:
        axes[1].plot(epochs, hist['val_accuracy'], 'r-o', label='Val Acc')
    axes[1].set_title(f'{phase_name} - Accuracy')
    axes[1].legend()
    axes[1].grid(True)
    
    idx = 2
    # 3. Plot F1 Score (nếu có)
    if has_f1:
        axes[idx].plot(epochs, hist['f1_score'], 'b-o', label='Train F1')
        if 'val_f1_score' in hist:
            axes[idx].plot(epochs, hist['val_f1_score'], 'r-o', label='Val F1')
        axes[idx].set_title(f'{phase_name} - F1 Score')
        axes[idx].legend()
        axes[idx].grid(True)
        idx += 1
        
    # 4. Plot Precision (nếu có)
    if has_prec:
        axes[idx].plot(epochs, hist['precision'], 'b-o', label='Train Prec')
        if 'val_precision' in hist:
            axes[idx].plot(epochs, hist['val_precision'], 'r-o', label='Val Prec')
        axes[idx].set_title(f'{phase_name} - Precision')
        axes[idx].legend()
        axes[idx].grid(True)
        idx += 1

    # 5. Plot Recall (nếu có)
    if has_rec:
        axes[idx].plot(epochs, hist['recall'], 'b-o', label='Train Recall')
        if 'val_recall' in hist:
            axes[idx].plot(epochs, hist['val_recall'], 'r-o', label='Val Recall')
        axes[idx].set_title(f'{phase_name} - Recall')
        axes[idx].legend()
        axes[idx].grid(True)
        idx += 1
    
    # Ẩn các ô trống nếu có
    while idx < len(axes):
        fig.delaxes(axes[idx])
        idx += 1
        
    plt.tight_layout()
    save_path = os.path.join(PLOT_PATH, f"History_{phase_name}.png")
    plt.savefig(save_path)
    plt.close()
    print(f"📈 Đã lưu biểu đồ Learning Curves tại: {save_path}")

def evaluate_phase(model, X, y_true, phase_name, target_names_list, is_multiclass=False):
    print(f"\n📊 [EVALUATING] Đang đánh giá giai đoạn: {phase_name}...")
    y_pred_raw = model.predict(X, verbose=1)
    if is_multiclass:
        y_pred = np.argmax(y_pred_raw, axis=1)
        avg_method = 'weighted'; cm_fmt = 'd'
        unique_labels = sorted(list(set(y_true) | set(y_pred)))
        valid_target_names = [target_names_list[i] for i in unique_labels]
    else:
        y_pred = (y_pred_raw > 0.5).astype(int).flatten()
        avg_method = 'binary'; cm_fmt = 'd'
        unique_labels = [0, 1]; valid_target_names = ['Normal', 'Attack']

    acc = accuracy_score(y_true, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average=avg_method, zero_division=0)
    
    print(f"\n>> KẾT QUẢ {phase_name.upper()}:")
    print(f"   Accuracy : {acc:.4f}")
    print(f"   Precision: {prec:.4f}")
    print(f"   Recall   : {rec:.4f}")
    print(f"   F1-Score : {f1:.4f}")
    
    report_str = classification_report(y_true, y_pred, target_names=valid_target_names, digits=4, labels=unique_labels, zero_division=0)
    print(report_str)

    report_file = os.path.join(REPORT_PATH, f"report_{phase_name}.txt")
    with open(report_file, "w", encoding='utf-8') as f:
        f.write(f"=== REPORT: {phase_name} ===\nAccuracy: {acc:.4f}\nPrecision: {prec:.4f}\nRecall: {rec:.4f}\nF1-Score: {f1:.4f}\n\n{report_str}")
    
    plt.figure(figsize=(10, 8))
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt=cm_fmt, cmap='Blues', xticklabels=valid_target_names, yticklabels=valid_target_names)
    plt.title(f"Confusion Matrix - {phase_name} (F1: {f1:.4f})")
    plt.savefig(os.path.join(PLOT_PATH, f"CM_{phase_name}.png")); plt.close()
    
    try:
        model.save(os.path.join(MODEL_PATH, f"model_{phase_name}.h5"), save_format='h5')
    except Exception as e: print(f"❌ Lỗi lưu model: {e}")

def plot_combined_timeline(history_df, drift_points):
    """
    Vẽ tất cả các đường (Acc, F1, Prec, Rec, Unc) trên một hình duy nhất
    """
    fig, ax1 = plt.subplots(figsize=(15, 8))

    # Trục Y bên trái: Các chỉ số hiệu năng (0 - 1)
    ax1.set_xlabel('Batch ID')
    ax1.set_ylabel('Performance Metrics', color='tab:blue')
    ln1 = ax1.plot(history_df['Batch_ID'], history_df['Accuracy'], label='Accuracy', color='blue', alpha=0.6, linewidth=1.5)
    ln2 = ax1.plot(history_df['Batch_ID'], history_df['F1-Score'], label='F1-Score', color='green', linewidth=2)
    ln3 = ax1.plot(history_df['Batch_ID'], history_df['Precision'], label='Precision', color='cyan', alpha=0.5, linestyle=':')
    ln4 = ax1.plot(history_df['Batch_ID'], history_df['Recall'], label='Recall', color='lime', alpha=0.5, linestyle=':')
    ax1.tick_params(axis='y', labelcolor='tab:blue')
    ax1.set_ylim(0, 1.05)
    ax1.grid(True, which='both', linestyle='--', linewidth=0.5)

    # Trục Y bên phải: Uncertainty & Threshold
    ax2 = ax1.twinx() 
    ax2.set_ylabel('Uncertainty / Threshold', color='tab:red')
    ln5 = ax2.plot(history_df['Batch_ID'], history_df['Uncertainty'], label='Uncertainty', color='red', linestyle='--', alpha=0.7)
    ln6 = ax2.plot(history_df['Batch_ID'], history_df['Threshold'], label='Dyn. Threshold', color='orange', linestyle='-.', alpha=0.8)
    ax2.tick_params(axis='y', labelcolor='tab:red')

    # Vẽ vạch Drift
    for d in drift_points:
        ax1.axvline(d, color='purple', alpha=0.3, linewidth=10, label='Drift Detected' if d == drift_points[0] else "")

    # Gộp Legend
    lines = ln1 + ln2 + ln3 + ln4 + ln5 + ln6
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='lower right', frameon=True, shadow=True)

    plt.title(f"Comprehensive Streaming Performance Timeline (All Metrics)\nTotal Batches: {len(history_df)}")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_PATH, "Combined_Performance_Timeline.png"))
    plt.close()
    print("✅ Đã lưu biểu đồ tổng hợp tại: Combined_Performance_Timeline.png")

# ==========================================
# 4. MAIN EXECUTION
# ==========================================
def main():
    print(">>> STARTING PURE CNN-ATTENTION SYSTEM (METRICS UPDATED) <<<")
    
    # 1. LOAD TRAIN DATA
    print("--- Loading Initial Data ---")
    df_init = pd.read_parquet(INITIAL_TRAIN_FILE)
    feat_cols = [c for c in df_init.columns if c not in ['Label', 'Label_Multi']]
    global FEATURE_NAMES; FEATURE_NAMES = feat_cols
    X_init = df_init[feat_cols].values; y_init_bin = df_init['Label'].values; y_init_multi = df_init['Label_Multi'].values

    X_seq, y_bin_seq = prepare_sequences(X_init, y_init_bin, TIME_STEPS)
    _, y_multi_seq = prepare_sequences(X_init, y_init_multi, TIME_STEPS)
    early = EarlyStopping(monitor='val_loss', patience=ES_PATIENCE, restore_best_weights=True)

    # ---------------------------------------------------------
    # GIAI ĐOẠN 1: BINARY TRAINING
    # ---------------------------------------------------------
    print("\n" + "="*40 + "\n🔰 PHASE 1: BINARY TRAINING (PURE CNN-ATTENTION)\n" + "="*40)
    model_A = create_binary_model((TIME_STEPS, len(feat_cols)))
    cw_A = class_weight.compute_class_weight('balanced', classes=np.unique(y_bin_seq), y=y_bin_seq)
    
    history_A = model_A.fit(X_seq, y_bin_seq, epochs=INITIAL_EPOCHS, batch_size=256, validation_split=0.1, callbacks=[early], class_weight=dict(enumerate(cw_A)), verbose=1)
    
    # [QUAN TRỌNG] Vẽ đồ thị History (5 hình) đúng format
    plot_training_history(history_A, "Phase1_Binary_Initial")
    
    evaluate_phase(model_A, X_seq, y_bin_seq, "Phase1_Binary_Initial", CLASS_NAMES, is_multiclass=False)

    buffer_A = SmartRehearsalBuffer(BUFFER_SIZE); buffer_A.add(X_seq, y_bin_seq, 0.0)

    # ---------------------------------------------------------
    # GIAI ĐOẠN 2: MULTICLASS TRAINING
    # ---------------------------------------------------------
    print("\n" + "="*40 + "\n🔰 PHASE 2: MULTICLASS TRAINING (PURE CNN-ATTENTION)\n" + "="*40)
    model_B = create_multiclass_model((TIME_STEPS, len(feat_cols)))
    idx_atk = np.where(y_multi_seq > 0)[0]
    buffer_B = MulticlassRehearsalBuffer(BUFFER_SIZE, N_CLASSES_MULTI)
    
    if len(idx_atk) > 0:
        X_B, y_B = X_seq[idx_atk], y_multi_seq[idx_atk]
        buffer_B.add(X_B, y_B)
        cw_B = class_weight.compute_class_weight('balanced', classes=np.unique(y_B), y=y_B)
        
        history_B = model_B.fit(X_B, y_B, epochs=INITIAL_EPOCHS, batch_size=256, validation_split=0.1, callbacks=[early], class_weight=dict(enumerate(cw_B)), verbose=1)
        plot_training_history(history_B, "Phase2_Multiclass_Initial")
        evaluate_phase(model_B, X_B, y_B, "Phase2_Multiclass_Initial", CLASS_NAMES, is_multiclass=True)
    else: print("⚠️ Không có mẫu tấn công trong tập Initial để train Multiclass!")

    # ---------------------------------------------------------
    # GIAI ĐOẠN 3: DRIFT STREAMING
    # ---------------------------------------------------------
    print("\n" + "="*40 + "\n🔰 PHASE 3: ONLINE STREAMING & DRIFT ADAPTATION\n" + "="*40)
    df_stream = pd.read_parquet(ONLINE_STREAM_FILE)
    X_st_raw = df_stream[feat_cols].values; y_st_bin = df_stream['Label'].values; y_st_multi = df_stream['Label_Multi'].values
    
    unc_hist = deque(maxlen=UNCERTAINTY_WINDOW)
    drift_points = []
    
    history_records = [] 
    
    y_true_stream_all, y_pred_stream_all = [], []
    y_true_m_stream, y_pred_m_stream = [], []
    
    controller = DynamicController(base_lr=1e-5, base_thresh=3.0)
    opt_A = keras.optimizers.Adam(learning_rate=1e-5); opt_B = keras.optimizers.Adam(learning_rate=1e-5)
    n_batches = len(X_st_raw) // RAW_BATCH_SIZE
    
    for i in range(n_batches):
        start = i * RAW_BATCH_SIZE; end = (i+1) * RAW_BATCH_SIZE
        X_curr = X_st_raw[start:end]
        X_sq, y_bin = prepare_sequences(X_curr, y_st_bin[start:end], TIME_STEPS)
        _, y_mul = prepare_sequences(X_curr, y_st_multi[start:end], TIME_STEPS)
        if len(X_sq) == 0: continue
        
        t0 = time.time(); p_A = model_A.predict(X_sq, verbose=0); t1 = time.time()
        latency = (t1 - t0) * 1000
        
        pred_bin = (p_A > 0.5).astype(int).flatten()
        
        acc = accuracy_score(y_bin, pred_bin)
        prec, rec, f1, _ = precision_recall_fscore_support(y_bin, pred_bin, average='binary', zero_division=0)
        
        unc = get_uncertainty(model_A, X_sq).numpy()
        
        y_true_stream_all.extend(y_bin); y_pred_stream_all.extend(pred_bin)
        
        idx_p = np.where(pred_bin == 1)[0]
        pred_m = np.zeros_like(y_mul)
        if len(idx_p) > 0:
            p_B = model_B.predict(X_sq[idx_p], verbose=0)
            pred_m[idx_p] = np.argmax(p_B, axis=1)
        y_true_m_stream.extend(y_mul); y_pred_m_stream.extend(pred_m)
        
        dyn_lr, dyn_thresh = controller.update(unc)
        opt_A.learning_rate.assign(dyn_lr); opt_B.learning_rate.assign(dyn_lr)
        
        is_drift = False
        if len(unc_hist) == UNCERTAINTY_WINDOW:
            if unc > np.mean(unc_hist) + dyn_thresh * np.std(unc_hist) and unc > 0.001: is_drift = True
        unc_hist.append(unc)
        
        history_records.append({
            'Batch_ID': i,
            'Accuracy': acc,
            'Precision': prec,
            'Recall': rec,
            'F1-Score': f1,
            'Uncertainty': unc,
            'Threshold': np.mean(unc_hist) + dyn_thresh * np.std(unc_hist) if len(unc_hist) > 0 else 0,
            'Is_Drift': 1 if is_drift else 0,
            'Learning_Rate': dyn_lr,
            'Latency_ms': latency
        })

        if is_drift:
            drift_points.append(i)
            print(f"⚡ [DRIFT DETECTED] Batch {i} | Unc: {unc:.4f} | F1: {f1:.4f}")
            if len(drift_points) == 1 or len(drift_points) % 10 == 0:
                bg = buffer_A.get_all_X(limit=20)
                if len(bg)>0: explain_drift_shap(model_A, bg, X_sq, i)
            
            X_old, y_old = buffer_A.get_sample(REHEARSAL_SAMPLE_SIZE)
            if len(X_old) > 0:
                X_ft = np.concatenate((X_sq, X_old)); y_ft = np.concatenate((y_bin, y_old))
                cw = class_weight.compute_class_weight('balanced', classes=np.unique(y_ft), y=y_ft)
                cw_d = dict(enumerate(cw))
                if 0 in cw_d: cw_d[0] *= NORMAL_CLASS_BOOST
                model_A.compile(optimizer=opt_A, loss='binary_crossentropy', metrics=['accuracy', tf.keras.metrics.Precision(name='precision'), tf.keras.metrics.Recall(name='recall'), F1Score()])
                for ep in range(ONLINE_EPOCHS): model_A.train_on_batch(X_ft, y_ft, class_weight=cw_d)
            
            idx_t = np.where(y_mul > 0)[0]
            if len(idx_t) > 0:
                X_old_B, y_old_B = buffer_B.get_sample(REHEARSAL_SAMPLE_SIZE)
                X_ft_B = np.concatenate((X_sq[idx_t], X_old_B)); y_ft_B = np.concatenate((y_mul[idx_t], y_old_B))
                model_B.compile(optimizer=opt_B, loss=SparseFocalLoss(), metrics=['accuracy'])
                for ep in range(ONLINE_EPOCHS): model_B.train_on_batch(X_ft_B, y_ft_B)
            unc_hist.clear()
            
        buffer_A.add(X_sq, y_bin, unc)
        if np.any(y_mul > 0): buffer_B.add(X_sq[y_mul > 0], y_mul[y_mul > 0])
        if i % 20 == 0: print(f"Batch {i}/{n_batches} | Acc: {acc:.4f} | F1: {f1:.4f} | Unc: {unc:.4f}")

    # ==========================================
    # 5. FINAL REPORT & SAVING
    # ==========================================
    print("\n" + "="*50 + "\n🏁 FINAL REPORT: SAVING DETAILED TABLES & PLOTS\n" + "="*50)
    
    df_history = pd.DataFrame(history_records)
    
    # [QUAN TRỌNG] LƯU FILE THEO ĐÚNG TÊN BẠN YÊU CẦU: Drift_Events_Detailed.csv
    csv_path_full = os.path.join(REPORT_PATH, "Drift_Events_Detailed.csv")
    df_history.to_csv(csv_path_full, index=False)
    print(f"✅ Đã lưu file chi tiết tại: {csv_path_full}")
    print(f"   (Bao gồm các cột: Batch_ID, Accuracy, F1-Score, Uncertainty, Threshold, Is_Drift, Learning_Rate, Latency_ms)")

    # Lưu thêm file chỉ chứa sự kiện Drift (tiện cho bạn kiểm tra nhanh)
    df_only_drift = df_history[df_history['Is_Drift'] == 1]
    if not df_only_drift.empty:
        csv_path_drift = os.path.join(REPORT_PATH, "Drift_Events_Only.csv")
        df_only_drift.to_csv(csv_path_drift, index=False)
        print(f"✅ Đã lưu file chỉ chứa các sự kiện Drift tại: {csv_path_drift}")

    plot_combined_timeline(df_history, drift_points)

    print("\n>> [STREAM BINARY RESULTS]")
    acc_s = accuracy_score(y_true_stream_all, y_pred_stream_all)
    prec_s, rec_s, f1_s, _ = precision_recall_fscore_support(y_true_stream_all, y_pred_stream_all, average='binary', zero_division=0)
    print(f"Accuracy : {acc_s:.4f}\nPrecision: {prec_s:.4f}\nRecall   : {rec_s:.4f}\nF1-Score : {f1_s:.4f}")
    
    with open(os.path.join(REPORT_PATH, "final_stream_report.txt"), "w", encoding='utf-8') as f:
        f.write("=== FINAL STREAMING REPORT ===\n\n")
        f.write(f"--- BINARY ---\nAcc: {acc_s:.4f}, Prec: {prec_s:.4f}, Rec: {rec_s:.4f}, F1: {f1_s:.4f}\n")

    plt.figure(figsize=(8, 6))
    sns.heatmap(confusion_matrix(y_true_stream_all, y_pred_stream_all), annot=True, fmt='d', cmap='Blues')
    plt.title("CM Stream Binary"); plt.savefig(os.path.join(PLOT_PATH, "CM_Phase3_Stream_Binary.png")); plt.close()
    
    # ==========================================
    # [ADD FOR COMPARISON] XUẤT DỮ LIỆU ĐỂ VẼ BIỂU ĐỒ SO SÁNH
    # ==========================================
    if 'history_records' in locals():
        df_compare = pd.DataFrame(history_records)
        # Đổi tên cột cho thống nhất với Master Script (nếu cần)
        # Ở file này cột tên là 'Accuracy', 'F1-Score', 'Latency_ms', 'Batch_ID'
        # Ta đổi tên lại cho chuẩn format chung
        df_compare = df_compare.rename(columns={
            'Batch_ID': 'batch',
            'Accuracy': 'accuracy',
            'F1-Score': 'f1',
            'Latency_ms': 'latency'
        })
        # Chỉ lấy các cột cần thiết
        df_compare = df_compare[['batch', 'accuracy', 'f1', 'latency']]
        
        save_path_cmp = os.path.join(PLOT_PATH, "history_cnn_attention_pure.csv")
        df_compare.to_csv(save_path_cmp, index=False)
        print(f"✅ [COMPARE] Đã lưu lịch sử Pure CNN Attention tại: {save_path_cmp}")
    
    print("DONE ALL PHASES.")

if __name__ == "__main__":
    main()