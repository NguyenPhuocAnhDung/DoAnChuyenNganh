import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model
from tensorflow.keras.callbacks import EarlyStopping
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, auc
from sklearn.utils import class_weight 
from collections import deque
import os
import time
import shap 
import random
import seaborn as sns

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# ==========================================
# 1. CẤU HÌNH
# ==========================================
DATA_PATH = "../../dataset/processed/processedstreamvs2.4"
PLOT_PATH = "../../baocao/main_cnn_gru_attention/plots" 
MODEL_PATH = "../../baocao/main_cnn_gru_attention/models"

if not os.path.exists(PLOT_PATH): os.makedirs(PLOT_PATH)
if not os.path.exists(MODEL_PATH): os.makedirs(MODEL_PATH)

INITIAL_TRAIN_FILE = os.path.join(DATA_PATH, "processed_initial_train_balanced.parquet")
ONLINE_STREAM_FILE = os.path.join(DATA_PATH, "processed_online_stream.parquet")

TIME_STEPS = 10 
BATCH_SIZE_INIT = 1024 
BATCH_SIZE_STREAM = 256 
RAW_BATCH_SIZE = BATCH_SIZE_STREAM * TIME_STEPS 
MC_SAMPLES = 10 
REHEARSAL_SAMPLE_SIZE = 4096 
BUFFER_SIZE = 20000 
ONLINE_EPOCHS = 3
LABELING_BUDGET = 0.05 
PSEUDO_CONFIDENCE = 0.95 
UNCERTAINTY_WINDOW = 50 
NORMAL_CLASS_BOOST = 1.0

CLASS_NAMES = ['Normal', 'DoS', 'PortScan', 'Botnet', 'BruteForce', 'WebAttack', 'Infiltration', 'DDoS']
NUM_CLASSES = len(CLASS_NAMES)
FEATURE_NAMES = [] 

# ==========================================
# 2. CUSTOM METRICS (ĐÃ SỬA LỖI TƯƠNG THÍCH)
# ==========================================
# [FIX] Dùng tf.keras.utils thay vì keras.saving để tương thích mọi phiên bản TF
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
        p = self.precision.result()
        r = self.recall.result()
        return 2 * ((p * r) / (p + r + 1e-7))
    def reset_state(self):
        self.precision.reset_state()
        self.recall.reset_state()

# ==========================================
# 3. ATTENTION & MODEL (CNN-GRU-ATTENTION)
# ==========================================
# [FIX] Dùng tf.keras.utils thay vì keras.saving
@tf.keras.utils.register_keras_serializable()
class AttentionBlock(layers.Layer):
    def __init__(self, **kwargs):
        super(AttentionBlock, self).__init__(**kwargs)
    def build(self, input_shape):
        self.W = self.add_weight(name='att_weight', shape=(input_shape[-1], 1), initializer='normal')
        self.b = self.add_weight(name='att_bias', shape=(input_shape[1], 1), initializer='zeros')
        super(AttentionBlock, self).build(input_shape)
    def call(self, x):
        e = keras.backend.tanh(keras.backend.dot(x, self.W) + self.b)
        a = keras.backend.softmax(e, axis=1)
        output = x * a
        return keras.backend.sum(output, axis=1)

def create_cnn_gru_attention(input_shape, num_classes=1):
    inputs = layers.Input(shape=input_shape)
    
    # 1. CNN Feature Extraction
    x = layers.Conv1D(128, 3, activation='relu', padding='same')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling1D(2)(x)
    class MCDropout(layers.Dropout):
        def call(self, i): return super().call(i, training=True)
    x = MCDropout(0.3)(x)
    
    # 2. Bi-GRU Sequential Learning
    x = layers.Bidirectional(layers.GRU(128, return_sequences=True))(x)
    x = MCDropout(0.3)(x)
    x = layers.Bidirectional(layers.GRU(64, return_sequences=True))(x)
    
    # 3. Attention Mechanism
    x = AttentionBlock()(x)
    
    # 4. Classifier
    x = layers.Dense(64, activation='relu')(x)
    x = MCDropout(0.3)(x)
    
    if num_classes == 1:
        outputs = layers.Dense(1, activation='sigmoid')(x)
    else:
        outputs = layers.Dense(num_classes, activation='softmax')(x)
    
    return Model(inputs, outputs, name="CNN_GRU_Attention")

# ==========================================
# 4. HELPERS (REPORTING & PLOTTING)
# ==========================================
def generate_admin_report(shap_values, feature_names, drift_data):
    if isinstance(shap_values, list): shap_values = shap_values[0]
    feature_importance = np.mean(np.abs(shap_values), axis=0)
    top_indices = np.argsort(feature_importance)[::-1][:3]
    
    print("\n" + "!"*50)
    print("🚨 [ADMIN REPORT] PHÂN TÍCH NGUYÊN NHÂN DRIFT 🚨")
    print("!"*50)
    
    top_cause = "Unknown"
    for i, idx in enumerate(top_indices):
        idx = int(idx)
        if feature_names and idx < len(feature_names): feat_name = feature_names[idx]
        else: feat_name = f"Feature_{idx}"
        
        val = feature_importance[idx]
        score = float(val.item()) if hasattr(val, 'item') else float(val)
        mean_val = float(np.mean(drift_data[:, idx]))
        
        print(f"👉 Tác nhân {i+1}: {feat_name} (Impact: {score:.4f}, Avg: {mean_val:.4f})")
        if i == 0: top_cause = feat_name
    print("-" * 40)
    return top_cause

def explain_drift_shap(model, background_data, drift_data, batch_idx):
    try:
        bg_2d = np.mean(background_data, axis=1)
        drift_2d = np.mean(drift_data[:20], axis=1) 
        def model_predict_wrapper(x_2d):
            x_3d = np.expand_dims(x_2d, axis=1).repeat(TIME_STEPS, axis=1)
            return model.predict(x_3d, verbose=0)
        explainer = shap.KernelExplainer(model_predict_wrapper, bg_2d[:10]) 
        shap_values = explainer.shap_values(drift_2d, nsamples=50, silent=True)
        return generate_admin_report(shap_values, FEATURE_NAMES, drift_2d)
    except Exception as e:
        print(f"⚠️ SHAP Error: {e}")
        return "Unknown (SHAP Error)"

def plot_history(history, model_name):
    metrics = ['loss', 'accuracy', 'f1_score', 'precision', 'recall']
    plt.figure(figsize=(20, 12))
    for i, metric in enumerate(metrics):
        if metric in history.history:
            plt.subplot(2, 3, i+1)
            plt.plot(history.history[metric], label=f'Train {metric}')
            if f'val_{metric}' in history.history:
                plt.plot(history.history[f'val_{metric}'], label=f'Val {metric}')
            plt.title(f'Model {metric.capitalize()}'); plt.legend(); plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_PATH, f"History_{model_name}.png")); plt.close()

def plot_evaluation_metrics(model, X_val, y_val, model_name="Binary"):
    print(f"📊 Drawing Confusion Matrix & ROC for {model_name}...")
    y_pred_prob = model.predict(X_val, verbose=0)
    y_pred = (y_pred_prob > 0.5).astype(int).flatten()
    
    cm = confusion_matrix(y_val, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Confusion Matrix - {model_name}'); plt.savefig(os.path.join(PLOT_PATH, f"CM_{model_name}.png")); plt.close()

    fpr, tpr, _ = roc_curve(y_val, y_pred_prob)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'AUC = {roc_auc:.4f}')
    plt.plot([0, 1], [0, 1], linestyle='--'); plt.legend()
    plt.title(f'ROC - {model_name}'); plt.savefig(os.path.join(PLOT_PATH, f"ROC_{model_name}.png")); plt.close()

def save_drift_report(drift_logs):
    if not drift_logs: return
    df = pd.DataFrame(drift_logs)
    cols = ['Drift_ID', 'Batch_Idx', 'Pre_Drift_Acc', 'Drop_Acc', 'Recovery_Batches', 'Primary_Cause', 'Uncertainty_Level']
    df = df[cols] if set(cols).issubset(df.columns) else df
    
    print("\n" + ">>" * 20)
    print("📊 BẢNG TỔNG HỢP CÁC SỰ KIỆN DRIFT")
    print(">>" * 20)
    print(df.to_markdown(index=False, floatfmt=".4f"))
    df.to_csv(os.path.join(PLOT_PATH, "Drift_Events_Detailed.csv"), index=False)

# === FUNCTIONS FOR FIGURE 5 & 6 ===
def save_figure_5_comparison(acc_with_al, acc_without_al):
    """
    Figure 5: So sánh Accuracy của model có Active Learning và không có (Static).
    """
    plt.figure(figsize=(12, 6))
    plt.plot(acc_with_al, label='With Active Learning', color='blue', linewidth=2)
    plt.plot(acc_without_al, label='Without Active Learning', color='gray', linestyle='--', linewidth=1.5, alpha=0.7)
    
    plt.title('Figure 5. Accuracy over time in the data stream (With vs Without Active Learning)')
    plt.xlabel('Batch Index')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.savefig(os.path.join(PLOT_PATH, "Figure5_Accuracy_Comparison.png"))
    plt.close()
    print(f"✅ Đã lưu Figure 5 tại: {PLOT_PATH}")

def save_figure_6_uncertainty(uncertainty_history, drift_logs):
    """
    Figure 6: Đường cong Uncertainty với vùng highlight drift.
    """
    plt.figure(figsize=(12, 6))
    plt.plot(uncertainty_history, label='Uncertainty Level', color='purple', linewidth=1)
    
    # Highlight Drift Intervals
    drift_indices = [d['Batch_Idx'] for d in drift_logs]
    for idx in drift_indices:
        plt.axvline(x=idx, color='red', linestyle='--', alpha=0.6)
        # Tô vùng +/- 2 batch xung quanh điểm drift
        start_shade = max(0, idx - 2)
        end_shade = idx + 2
        plt.axvspan(start_shade, end_shade, color='red', alpha=0.2, label='Drift Interval' if idx == drift_indices[0] else "")

    plt.title('Figure 6. Uncertainty curve across stream batches highlighting drift intervals')
    plt.xlabel('Batch Index')
    plt.ylabel('Uncertainty')
    plt.legend()
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.savefig(os.path.join(PLOT_PATH, "Figure6_Uncertainty_Curve.png"))
    plt.close()
    print(f"✅ Đã lưu Figure 6 tại: {PLOT_PATH}")

# ==========================================
# 5. CLASSES (ACTIVE LEARNING)
# ==========================================
class ActiveStrategy:
    def __init__(self, budget_ratio=0.2): self.budget_ratio = budget_ratio
    def select_samples(self, X_batch, y_true, y_prob, unc):
        n = len(X_batch); n_bud = int(n * self.budget_ratio)
        idx = np.argsort(unc); q_idx = idx[-n_bud:]; rem_idx = idx[:-n_bud]
        conf = np.abs(y_prob[rem_idx] - 0.5) * 2
        p_idx = rem_idx[conf > PSEUDO_CONFIDENCE]
        X_p, y_p = X_batch[p_idx], (y_prob[p_idx] > 0.5).astype(int).flatten()
        X_q, y_q = X_batch[q_idx], y_true[q_idx]
        if len(X_p) > 0: return np.concatenate([X_q, X_p]), np.concatenate([y_q, y_p])
        else: return X_q, y_q

class DynamicController:
    def __init__(self, base_lr=1e-5): self.base_lr = base_lr; self.ema_uncertainty = 0.01 
    def update(self, current_uncertainty):
        self.ema_uncertainty = 0.9 * self.ema_uncertainty + 0.1 * current_uncertainty
        ratio = current_uncertainty / (self.ema_uncertainty + 1e-9)
        return min(self.base_lr * max(1.0, ratio * 20.0), 0.001)

class SmartRehearsalBuffer:
    def __init__(self, max_size): self.max_size = max_size; self.buffer = [] 
    def add(self, x, y, unc):
        for i in range(len(x)): self.buffer.append((x[i], y[i], unc))
        self.buffer.sort(key=lambda x: x[2], reverse=True)
        if len(self.buffer) > self.max_size:
            n_h = int(self.max_size*0.6); self.buffer = self.buffer[:n_h] + random.sample(self.buffer[n_h:], self.max_size-n_h)
    def get_sample(self, n):
        if not self.buffer: return np.array([]), np.array([])
        s = random.sample(self.buffer, min(len(self.buffer), n))
        return np.array([x[0] for x in s]), np.array([x[1] for x in s])
    def get_all_X(self, limit=50):
        if not self.buffer: return np.array([])
        return np.array([x[0] for x in random.sample(self.buffer, min(len(self.buffer), limit))])

def prepare_sequences(X, y, time_steps):
    if len(X) < time_steps: return np.array([]), np.array([])
    n = len(X) // time_steps
    X_seq = X[:n*time_steps].reshape((n, time_steps, X.shape[1]))
    y_seq = y[time_steps-1::time_steps]
    return X_seq, y_seq

@tf.function(reduce_retracing=True)
def get_uncertainty(model, x_batch):
    preds = tf.stack([model(x_batch, training=False) for _ in range(MC_SAMPLES)], axis=0)
    return tf.reduce_mean(tf.math.reduce_variance(preds, axis=0))

# ==========================================
# 6. MAIN EXECUTION
# ==========================================
def main():
    print(">>> RUNNING: CNN - GRU - ATTENTION + ACTIVE LEARNING <<<")
    
    df_init = pd.read_parquet(INITIAL_TRAIN_FILE)
    feat_cols = [c for c in df_init.columns if c not in ['Label', 'Label_Multi']]
    global FEATURE_NAMES; FEATURE_NAMES = feat_cols
    n_features = len(feat_cols)
    X_init = df_init[feat_cols].values; y_init = df_init['Label'].values
    X_seq, y_seq = prepare_sequences(X_init, y_init, TIME_STEPS)

    # 1. Init Model
    model = create_cnn_gru_attention((TIME_STEPS, n_features), num_classes=1)
    
    # Metrics đầy đủ
    my_metrics = ['accuracy', F1Score(name='f1_score'), tf.keras.metrics.Precision(name='precision'), tf.keras.metrics.Recall(name='recall')]
    lr_schedule = tf.keras.optimizers.schedules.CosineDecay(1e-3, 20000)
    model.compile(optimizer=tf.keras.optimizers.AdamW(learning_rate=lr_schedule, weight_decay=1e-4), 
                  loss='binary_crossentropy', metrics=my_metrics)
    
    # 2. Train Phase 1
    print("\n🚀 GIAI ĐOẠN 1: HUẤN LUYỆN BINARY")
    early = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    cw = class_weight.compute_class_weight('balanced', classes=np.unique(y_seq), y=y_seq)
    
    history = model.fit(X_seq, y_seq, epochs=100, batch_size=BATCH_SIZE_INIT, validation_split=0.1, 
                        callbacks=[early], class_weight=dict(enumerate(cw)), verbose=1)
    
    plot_history(history, "CNN_GRU_Attention_Binary")
    plot_evaluation_metrics(model, X_seq, y_seq, "Binary_Offline")
    
    # [NEW] Tạo Model Static
    print("\n📝 Creating Static Model for Comparison (Figure 5)...")
    # Clone model đã fix lỗi serializable
    model_static = keras.models.clone_model(model)
    model_static.set_weights(model.get_weights())
    model_static.compile(loss='binary_crossentropy', metrics=['accuracy']) 

    # 3. Setup Streaming
    buffer = SmartRehearsalBuffer(BUFFER_SIZE); buffer.add(X_seq, y_seq, 0.0)
    
    print("\n🌊 GIAI ĐOẠN 3: STREAMING & DRIFT MONITORING")
    # Streaming compile cho dynamic model
    model.compile(optimizer=keras.optimizers.AdamW(learning_rate=1e-5, weight_decay=1e-4), 
                  loss='binary_crossentropy', metrics=['accuracy'])
    
    df_stream = pd.read_parquet(ONLINE_STREAM_FILE)
    X_st = df_stream[feat_cols].values; y_st = df_stream['Label'].values
    
    unc_hist = deque(maxlen=UNCERTAINTY_WINDOW)
    
    metrics = {'acc': [], 'lat': [], 'unc': [], 'f1': [], 'acc_static': []}
    
    active_strategy = ActiveStrategy(budget_ratio=LABELING_BUDGET)
    controller = DynamicController(base_lr=1e-5)
    
    drift_logs = []
    active_recovery = None 
    baseline_acc = 0.95 
    acc_window = deque(maxlen=10)
    
    n_batches = len(X_st) // RAW_BATCH_SIZE
    for i in range(n_batches):
        start = i * RAW_BATCH_SIZE; end = (i+1) * RAW_BATCH_SIZE
        X_curr = X_st[start:end]; X_sq, y_b = prepare_sequences(X_curr, y_st[start:end], TIME_STEPS)
        if len(X_sq) == 0: continue

        t_start = time.time()
        
        # 1. Predict Dynamic Model
        p = model.predict(X_sq, verbose=0)
        y_pred = (p > 0.5).astype(int).flatten()
        acc = accuracy_score(y_b, y_pred)
        f1 = f1_score(y_b, y_pred, zero_division=0)
        unc = get_uncertainty(model, X_sq).numpy()
        
        # 2. Predict Static Model
        p_static = model_static.predict(X_sq, verbose=0)
        y_pred_static = (p_static > 0.5).astype(int).flatten()
        acc_static = accuracy_score(y_b, y_pred_static)
        
        lat = (time.time() - t_start) * 1000
        
        metrics['acc'].append(acc)
        metrics['acc_static'].append(acc_static)
        metrics['lat'].append(lat)
        metrics['unc'].append(unc)
        metrics['f1'].append(f1)
        
        # Update Baseline
        acc_window.append(acc)
        current_baseline = np.mean(acc_window) if len(acc_window) > 0 else acc
        
        dyn_lr = controller.update(unc)
        model.optimizer.learning_rate = dyn_lr
        
        unc_hist.append(unc)
        is_drift = False
        if len(unc_hist) == UNCERTAINTY_WINDOW and unc > np.mean(unc_hist) + 3 * np.std(unc_hist) and unc > 0.001:
            is_drift = True
        
        # DRIFT HANDLING
        if is_drift:
            if active_recovery:
                drift_logs[-1]['Recovery_Batches'] = -1; active_recovery = None
                
            print(f"⚡ [DRIFT DETECTED] Batch {i} | Unc: {unc:.4f} | Drop Acc: {acc:.4f}")
            
            X_act, y_act = active_strategy.select_samples(X_sq, y_b, p.flatten(), unc)
            bg = buffer.get_all_X(20)
            cause = explain_drift_shap(model, bg, X_sq, i) if len(bg) > 0 else "N/A"
            
            drift_logs.append({
                'Drift_ID': len(drift_logs)+1, 'Batch_Idx': i, 'Pre_Drift_Acc': current_baseline,
                'Drop_Acc': acc, 'Recovery_Batches': np.nan, 'Primary_Cause': cause, 'Uncertainty_Level': unc
            })
            active_recovery = {'start_idx': i, 'target_acc': current_baseline * 0.98}
            
            X_old, y_old = buffer.get_sample(REHEARSAL_SAMPLE_SIZE)
            if len(X_old) > 0:
                X_ft = np.concatenate((X_act, X_old)); y_ft = np.concatenate((y_act, y_old))
                cw_ft = class_weight.compute_class_weight('balanced', classes=np.unique(y_ft), y=y_ft)
                cw_dict = dict(enumerate(cw_ft)); cw_dict[0] *= NORMAL_CLASS_BOOST
                
                best_loss = float('inf'); pat = 0
                for _ in range(ONLINE_EPOCHS):
                    l = model.train_on_batch(X_ft, y_ft, class_weight=cw_dict)
                    if isinstance(l, list): l=l[0]
                    if l < best_loss - 0.001: best_loss = l; pat = 0
                    else: pat += 1
                    if pat >= 3: break
            unc_hist.clear()
        
        # RECOVERY CHECK
        if active_recovery:
            if acc >= active_recovery['target_acc']:
                rec = i - active_recovery['start_idx']
                drift_logs[-1]['Recovery_Batches'] = rec
                print(f"✅ Recovered in {rec} batches")
                active_recovery = None
            elif (i - active_recovery['start_idx']) > 200:
                drift_logs[-1]['Recovery_Batches'] = -1; active_recovery = None

        buffer.add(X_sq, y_b, unc)
        if i % 20 == 0: 
            print(f"Batch {i} | Acc: {acc:.4f} | Static: {acc_static:.4f} | Lat: {lat:.2f}ms")

    # Final Report
    print(f"\nAvg Accuracy (Dynamic): {np.mean(metrics['acc']):.4f}")
    
    save_figure_5_comparison(metrics['acc'], metrics['acc_static'])
    save_figure_6_uncertainty(metrics['unc'], drift_logs)
    save_drift_report(drift_logs)
    
    model.save(os.path.join(MODEL_PATH, "CNN_GRU_Attention.h5"))
    
    # Export Compare Data
    if 'metrics' in locals():
        df_compare = pd.DataFrame({
            'batch': range(len(metrics['acc'])),
            'accuracy': metrics['acc'],
            'accuracy_static': metrics['acc_static'],
            'f1': metrics['f1'],
            'latency': metrics['lat']
        })
        save_path_cmp = os.path.join(PLOT_PATH, "history_cnn_gru_attention.csv")
        df_compare.to_csv(save_path_cmp, index=False)
        print(f"✅ [COMPARE] Saved history to: {save_path_cmp}")
        
    print("✅ Done.")

if __name__ == "__main__":
    main()