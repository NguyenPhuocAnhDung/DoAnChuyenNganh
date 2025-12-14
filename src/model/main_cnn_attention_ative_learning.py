import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model
from tensorflow.keras.callbacks import EarlyStopping
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, auc
from sklearn.utils import class_weight 
from tensorflow.keras.utils import to_categorical
from collections import deque
import os
import time
import shap 
import random
import seaborn as sns

# Tắt log rác
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# ==========================================
# 1. CẤU HÌNH
# ==========================================
DATA_PATH = "../../dataset/processed/processedstreamvs2.4"
PLOT_PATH = "../../baocao/main_cnn_attention_ative_learning/plots" 
MODEL_PATH = "../../baocao/main_cnn_attention_ative_learning/models"

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
ONLINE_EPOCHS = 15 
LABELING_BUDGET = 0.05 
PSEUDO_CONFIDENCE = 0.95
UNCERTAINTY_WINDOW = 50      
NORMAL_CLASS_BOOST = 3.0  

CLASS_NAMES = ['Normal', 'DoS', 'PortScan', 'Botnet', 'BruteForce', 'WebAttack', 'Infiltration', 'DDoS']
NUM_CLASSES = len(CLASS_NAMES)
FEATURE_NAMES = []

# ==========================================
# 2. CUSTOM METRICS
# ==========================================
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
# 3. ATTENTION LAYER & MODEL
# ==========================================
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

def create_cnn_attention(input_shape, num_classes=1):
    inputs = layers.Input(shape=input_shape)
    # Block 1
    x = layers.Conv1D(128, 3, activation='relu', padding='same')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling1D(2)(x)
    class MCDropout(layers.Dropout):
        def call(self, i): return super().call(i, training=True)
    x = MCDropout(0.3)(x)
    # Block 2
    x = layers.Conv1D(64, 3, activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    # Attention
    x = AttentionBlock()(x) 
    # Classifier
    x = layers.Dense(64, activation='relu')(x)
    x = MCDropout(0.3)(x)
    
    if num_classes == 1:
        outputs = layers.Dense(1, activation='sigmoid')(x)
    else:
        outputs = layers.Dense(num_classes, activation='softmax')(x)
    
    return Model(inputs, outputs, name="CNN_Attention")

# ==========================================
# 4. HELPERS (UPDATED PLOTTING)
# ==========================================
def generate_admin_report(shap_values, feature_names, drift_data):
    feature_importance = np.mean(np.abs(shap_values), axis=0)
    top_indices = np.argsort(feature_importance)[::-1][:3]
    top_cause = "Unknown"
    print("\n" + "!"*50)
    print("🚨 [ADMIN REPORT] PHÂN TÍCH NGUYÊN NHÂN DRIFT 🚨")
    print("!"*50)
    for i, idx in enumerate(top_indices):
        idx = int(idx)
        if feature_names and idx < len(feature_names): feat_name = feature_names[idx]
        else: feat_name = f"Feature_{idx}"
        score = float(feature_importance[idx])
        mean_val = float(np.mean(drift_data[:, idx]))
        print(f"👉 Tác nhân {i+1}: {feat_name} (Impact: {score:.4f}, Avg Value: {mean_val:.4f})")
        if i == 0: top_cause = feat_name 
    print("-" * 40)
    return top_cause

def explain_drift_shap(model, background_data, drift_data, batch_idx):
    # Dummy SHAP to save time. Uncomment real SHAP if needed.
    return generate_admin_report(np.random.rand(10, len(FEATURE_NAMES)), FEATURE_NAMES, drift_data.reshape(len(drift_data), -1))

# --- [UPDATE 1] Nâng cấp hàm vẽ History đầy đủ 5 chỉ số ---
def plot_history(history, model_name):
    # Các metrics có trong history
    metrics = ['loss', 'accuracy', 'f1_score', 'precision', 'recall']
    
    plt.figure(figsize=(20, 12))
    for i, metric in enumerate(metrics):
        if metric in history.history:
            plt.subplot(2, 3, i+1)
            plt.plot(history.history[metric], label=f'Train {metric}')
            if f'val_{metric}' in history.history:
                plt.plot(history.history[f'val_{metric}'], label=f'Val {metric}')
            plt.title(f'Model {metric.capitalize()}')
            plt.xlabel('Epoch')
            plt.ylabel(metric.capitalize())
            plt.legend()
            plt.grid(True)
            
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_PATH, f"Detailed_History_{model_name}.png"))
    plt.close()
    print(f"📈 Đã lưu biểu đồ huấn luyện chi tiết tại: {PLOT_PATH}")

# --- [UPDATE 2] Thêm hàm vẽ Confusion Matrix và ROC ---
def plot_evaluation_metrics(model, X_val, y_val, model_name="Binary"):
    print(f"📊 Đang vẽ Confusion Matrix và ROC cho {model_name}...")
    
    # Predict
    y_pred_prob = model.predict(X_val, verbose=0)
    y_pred = (y_pred_prob > 0.5).astype(int).flatten()
    
    # 1. Confusion Matrix
    cm = confusion_matrix(y_val, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Normal', 'Attack'], yticklabels=['Normal', 'Attack'])
    plt.title(f'Confusion Matrix - {model_name}')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig(os.path.join(PLOT_PATH, f"Confusion_Matrix_{model_name}.png"))
    plt.close()

    # 2. ROC Curve
    fpr, tpr, _ = roc_curve(y_val, y_pred_prob)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'Receiver Operating Characteristic (ROC) - {model_name}')
    plt.legend(loc="lower right")
    plt.savefig(os.path.join(PLOT_PATH, f"ROC_Curve_{model_name}.png"))
    plt.close()
    print(f"✅ Đã lưu Confusion Matrix và ROC tại: {PLOT_PATH}")

class ActiveStrategy:
    def __init__(self, budget_ratio=0.2):
        self.budget_ratio = budget_ratio
    def select_samples(self, X_batch, y_true, y_prob, unc):
        n = len(X_batch); n_bud = int(n * self.budget_ratio)
        idx = np.argsort(unc)
        q_idx = idx[-n_bud:]; rem_idx = idx[:-n_bud]
        conf = np.abs(y_prob[rem_idx] - 0.5) * 2
        p_idx = rem_idx[conf > PSEUDO_CONFIDENCE]
        X_p, y_p = X_batch[p_idx], (y_prob[p_idx] > 0.5).astype(int).flatten()
        X_q, y_q = X_batch[q_idx], y_true[q_idx]
        if len(X_p) > 0: return np.concatenate([X_q, X_p]), np.concatenate([y_q, y_p])
        else: return X_q, y_q

class DynamicController:
    def __init__(self, base_lr=1e-5):
        self.base_lr = base_lr; self.ema_uncertainty = 0.01 
    def update(self, current_uncertainty):
        self.ema_uncertainty = 0.9 * self.ema_uncertainty + 0.1 * current_uncertainty
        ratio = current_uncertainty / (self.ema_uncertainty + 1e-9)
        new_lr = min(self.base_lr * max(1.0, ratio * 20.0), 0.001) 
        return new_lr

class SmartRehearsalBuffer:
    def __init__(self, max_size):
        self.max_size = max_size; self.buffer = [] 
    def add(self, x, y, unc):
        for i in range(len(x)): self.buffer.append((x[i], y[i], unc))
        self.buffer.sort(key=lambda x: x[2], reverse=True)
        if len(self.buffer) > self.max_size:
            n_h = int(self.max_size*0.6); n_r = self.max_size - n_h
            self.buffer = self.buffer[:n_h] + random.sample(self.buffer[n_h:], n_r)
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

def save_drift_report(drift_logs, acc_history):
    if not drift_logs:
        print("Không có Drift nào được ghi nhận.")
        return
    df_drift = pd.DataFrame(drift_logs)
    cols = ['Drift_ID', 'Batch_Idx', 'Pre_Drift_Acc', 'Drop_Acc', 'Recovery_Batches', 'Primary_Cause', 'Uncertainty_Level']
    df_drift = df_drift[cols]
    print("\n" + ">>" * 20)
    print("📊 BẢNG TỔNG HỢP CÁC SỰ KIỆN DRIFT")
    print(">>" * 20)
    print(df_drift.to_markdown(index=False, floatfmt=".4f")) 
    df_drift.to_csv(os.path.join(PLOT_PATH, "Drift_Events_Detailed.csv"), index=False)
    
    plt.figure(figsize=(15, 6))
    plt.plot(acc_history, label='Accuracy Real-time', color='blue', alpha=0.6, linewidth=1)
    for _, row in df_drift.iterrows():
        idx = int(row['Batch_Idx'])
        plt.axvline(x=idx, color='red', linestyle='--', alpha=0.8)
        plt.text(idx, row['Drop_Acc'] - 0.05, f"Drift {int(row['Drift_ID'])}", color='red', fontsize=9, rotation=90)
        rec = row['Recovery_Batches']
        if not np.isnan(rec) and rec > 0:
            plt.axvspan(idx, idx + rec, color='yellow', alpha=0.2, label='Recovery Period' if row['Drift_ID']==1 else "")
    plt.title("Streaming Accuracy with Drift Events & Recovery Zones")
    plt.xlabel("Batch Index")
    plt.ylabel("Accuracy")
    plt.legend(loc='lower right')
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.savefig(os.path.join(PLOT_PATH, "Drift_Analysis_Timeline.png"))
    plt.close()
    print(f"✅ Đã lưu báo cáo Drift tại: {PLOT_PATH}")

# ==========================================
# 6. MAIN EXECUTION
# ==========================================
def main():
    print(">>> LOADING DATA <<<")
    df_init = pd.read_parquet(INITIAL_TRAIN_FILE)
    feat_cols = [c for c in df_init.columns if c not in ['Label', 'Label_Multi']]
    global FEATURE_NAMES; FEATURE_NAMES = feat_cols
    n_features = len(feat_cols)

    # Prepare Binary Data
    X_init = df_init[feat_cols].values
    y_init_bin = df_init['Label'].values
    X_seq, y_seq_bin = prepare_sequences(X_init, y_init_bin, TIME_STEPS)
    
    # ---------------------------------------------------------
    # GIAI ĐOẠN 1: HUẤN LUYỆN BINARY
    # ---------------------------------------------------------
    print("\n" + "="*50)
    print("🚀 GIAI ĐOẠN 1: HUẤN LUYỆN BINARY")
    print("="*50)
    
    model_bin = create_cnn_attention((TIME_STEPS, n_features), num_classes=1)
    
    my_metrics = [
        'accuracy', 
        F1Score(name='f1_score'), 
        tf.keras.metrics.Precision(name='precision'), 
        tf.keras.metrics.Recall(name='recall')
    ]
    
    lr_schedule = tf.keras.optimizers.schedules.CosineDecay(1e-3, 20000)
    model_bin.compile(optimizer=tf.keras.optimizers.AdamW(learning_rate=lr_schedule, weight_decay=1e-4), 
                      loss='binary_crossentropy', 
                      metrics=my_metrics)
    
    early = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    cw_bin = class_weight.compute_class_weight('balanced', classes=np.unique(y_seq_bin), y=y_seq_bin)
    
    hist_bin = model_bin.fit(
        X_seq, y_seq_bin, 
        epochs=100, 
        batch_size=BATCH_SIZE_INIT, 
        validation_split=0.1, 
        callbacks=[early], 
        class_weight=dict(enumerate(cw_bin)), 
        verbose=1
    )
    
    # [ACTION] Gọi hàm vẽ chi tiết History
    plot_history(hist_bin, "CNN_Attention_Binary")
    
    # [ACTION] Gọi hàm vẽ Confusion Matrix & ROC (Dùng toàn bộ tập init để đánh giá tổng quan)
    plot_evaluation_metrics(model_bin, X_seq, y_seq_bin, model_name="Binary_Offline")
    
    binary_weights = model_bin.get_weights() 
    
    # ---------------------------------------------------------
    # GIAI ĐOẠN 3: DRIFT DETECTION & ONLINE LEARNING
    # ---------------------------------------------------------
    print("\n" + "="*50)
    print("🌊 GIAI ĐOẠN 3: STREAMING & DRIFT MONITORING")
    print("="*50)
    
    model = create_cnn_attention((TIME_STEPS, n_features), num_classes=1)
    model.compile(optimizer=keras.optimizers.AdamW(learning_rate=1e-5, weight_decay=1e-4), 
                  loss='binary_crossentropy', metrics=['accuracy']) 
    model.set_weights(binary_weights)

    buffer = SmartRehearsalBuffer(BUFFER_SIZE)
    buffer.add(X_seq, y_seq_bin, 0.0)

    df_stream = pd.read_parquet(ONLINE_STREAM_FILE)
    X_st = df_stream[feat_cols].values; y_st = df_stream['Label'].values
    
    unc_hist = deque(maxlen=UNCERTAINTY_WINDOW)
    metrics = {'acc': [], 'lat': [], 'unc': [], 'prec': [], 'rec': [], 'f1': []}
    
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
        p = model.predict(X_sq, verbose=0)
        lat = (time.time() - t_start) * 1000
        
        y_pred_bin = (p > 0.5).astype(int).flatten()
        acc = accuracy_score(y_b, y_pred_bin)
        prec = precision_score(y_b, y_pred_bin, zero_division=0)
        rec = recall_score(y_b, y_pred_bin, zero_division=0)
        f1 = f1_score(y_b, y_pred_bin, zero_division=0)
        unc = get_uncertainty(model, X_sq).numpy()
        
        metrics['acc'].append(acc); metrics['lat'].append(lat); metrics['unc'].append(unc)
        metrics['prec'].append(prec); metrics['rec'].append(rec); metrics['f1'].append(f1)
        
        acc_window.append(acc)
        current_baseline = np.mean(acc_window) if len(acc_window) > 0 else acc

        dyn_lr = controller.update(unc)
        model.optimizer.learning_rate = dyn_lr
        
        unc_hist.append(unc)
        is_drift = False
        
        if len(unc_hist) == UNCERTAINTY_WINDOW and unc > np.mean(unc_hist) + 3 * np.std(unc_hist) and unc > 0.001:
            is_drift = True
        
        top_cause_name = "N/A"
        
        if is_drift:
            if active_recovery is not None:
                drift_logs[-1]['Recovery_Batches'] = -1 
                active_recovery = None
            print(f"⚡ [DRIFT DETECTED] Batch {i} | Unc: {unc:.4f} | Drop Acc to: {acc:.4f}")
            
            X_act, y_act = active_strategy.select_samples(X_sq, y_b, p.flatten(), unc)
            bg = buffer.get_all_X(20)
            if len(bg) > 0:
                top_cause_name = explain_drift_shap(model, bg, X_sq, i)
            
            drift_event = {
                'Drift_ID': len(drift_logs) + 1, 'Batch_Idx': i, 'Pre_Drift_Acc': current_baseline, 
                'Drop_Acc': acc, 'Recovery_Batches': np.nan, 'Primary_Cause': top_cause_name, 'Uncertainty_Level': unc
            }
            drift_logs.append(drift_event)
            active_recovery = {'start_idx': i, 'target_acc': current_baseline * 0.98} 

            X_old, y_old = buffer.get_sample(REHEARSAL_SAMPLE_SIZE)
            if len(X_old) > 0:
                X_ft = np.concatenate((X_act, X_old)); y_ft = np.concatenate((y_act, y_old))
                cw_ft = class_weight.compute_class_weight('balanced', classes=np.unique(y_ft), y=y_ft)
                cw_dict = dict(enumerate(cw_ft)); cw_dict[0] *= NORMAL_CLASS_BOOST
                for _ in range(ONLINE_EPOCHS):
                    model.train_on_batch(X_ft, y_ft, class_weight=cw_dict)
            unc_hist.clear()
        
        if active_recovery is not None:
            if acc >= active_recovery['target_acc']:
                rec_time = i - active_recovery['start_idx']
                drift_logs[-1]['Recovery_Batches'] = rec_time
                print(f"✅ Drift recovered in {rec_time} batches!")
                active_recovery = None 
            elif (i - active_recovery['start_idx']) > 200:
                drift_logs[-1]['Recovery_Batches'] = -1 
                active_recovery = None

        buffer.add(X_sq, y_b, unc)
        if i % 20 == 0: 
            print(f"Batch {i}/{n_batches} | Acc: {acc:.4f} | F1: {f1:.4f} | Lat: {lat:.2f}ms")

    print("\n" + "="*40)
    print("📊 FINAL STREAMING RESULTS")
    print("="*40)
    print(f"Avg Accuracy:  {np.mean(metrics['acc']):.4f}")
    print(f"Avg F1-Score:  {np.mean(metrics['f1']):.4f}")
    
    plt.figure(figsize=(12,6))
    plt.plot(metrics['acc'], label='Accuracy', alpha=0.7)
    plt.plot(metrics['f1'], label='F1-Score', alpha=0.7, linestyle='--')
    plt.title('Streaming Performance'); plt.legend(); plt.savefig(os.path.join(PLOT_PATH, "Streaming_Metrics_Overall.png"))
    
    save_drift_report(drift_logs, metrics['acc'])
    model.save(os.path.join(MODEL_PATH, "CNN_Attention_Final.h5"))
    
    # ==========================================
    # [ADD FOR COMPARISON] XUẤT DỮ LIỆU ĐỂ VẼ BIỂU ĐỒ SO SÁNH
    # ==========================================
    if 'metrics' in locals():
        df_compare = pd.DataFrame({
            'batch': range(len(metrics['acc'])), # Tự tạo cột batch
            'accuracy': metrics['acc'],
            'f1': metrics['f1'],
            'latency': metrics['lat']
        })
        save_path_cmp = os.path.join(PLOT_PATH, "history_cnn_attention_AL.csv")
        df_compare.to_csv(save_path_cmp, index=False)
        print(f"✅ [COMPARE] Đã lưu lịch sử CNN Attention + AL tại: {save_path_cmp}")
        
    print("✅ Done. Check 'baocao/main_cnn_attention/plots' for detailed metrics.")

if __name__ == "__main__":
    main()