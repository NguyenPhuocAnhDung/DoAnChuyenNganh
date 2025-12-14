import json
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from kafka import KafkaConsumer
import time
import os
import csv
from datetime import datetime
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from collections import deque
import random

# ==========================================
# 1. CẤU HÌNH
# ==========================================
# MODEL_PATH = "D:/DACN/results/models/CNN_GRU_Attention.h5"
# MODEL_PATH = "D:/DACN/results/models/CNN_Attention.h5"
MODEL_PATH = "D:/DACN/results/models/final_model_A.h5"
# MODEL_PATH = "D:/DACN/results/models/transformer_sota_A.keras"
# MODEL_PATH = "D:/DACN/results/models/main_active_learning.h5"
TOPIC = "nids-traffic"
BATCH_SIZE = 256
LOG_FILE = "active_learning_final_model_A_log.csv"
PLOT_FILE = "demo_performance_final_model_A_active.png"

# Cấu hình Active Learning
LABELING_BUDGET = 0.05   
PSEUDO_CONFIDENCE = 0.95
UNCERTAINTY_WINDOW = 50      
NORMAL_CLASS_BOOST = 3.0 
MC_SAMPLES = 5 
REHEARSAL_SIZE = 2000 

CLASS_NAMES = ['Normal', 'DoS', 'PortScan', 'Botnet', 'BruteForce', 'WebAttack', 'Infiltration', 'DDoS']

# ==========================================
# 2. CUSTOM LAYERS & HELPERS
# ==========================================
class AttentionBlock(layers.Layer):
    def __init__(self, **kwargs): super(AttentionBlock, self).__init__(**kwargs)
    def build(self, input_shape):
        self.W = self.add_weight(name='att_weight', shape=(input_shape[-1], 1), initializer='normal')
        self.b = self.add_weight(name='att_bias', shape=(input_shape[1], 1), initializer='zeros')
        super(AttentionBlock, self).build(input_shape)
    def call(self, x):
        e = keras.backend.tanh(keras.backend.dot(x, self.W) + self.b)
        a = keras.backend.softmax(e, axis=1)
        output = x * a
        return keras.backend.sum(output, axis=1)
    def get_config(self): return super().get_config()

class MCDropout(layers.Dropout):
    def call(self, inputs): return super().call(inputs, training=True)
    def get_config(self): return super().get_config()

@tf.function(reduce_retracing=True)
def get_uncertainty(model, x_batch):
    preds = tf.stack([model(x_batch, training=False) for _ in range(MC_SAMPLES)], axis=0)
    return tf.reduce_mean(tf.math.reduce_variance(preds, axis=0))

class ActiveStrategy:
    def __init__(self, budget_ratio=0.05):
        self.budget_ratio = budget_ratio
    def select_samples(self, X_batch, y_true, y_prob, unc):
        n = len(X_batch); n_bud = int(n * self.budget_ratio)
        idx = np.argsort(unc)
        q_idx = idx[-n_bud:] # Mẫu khó
        X_q, y_q = X_batch[q_idx], y_true[q_idx]
        
        rem_idx = idx[:-n_bud]
        conf = np.abs(y_prob[rem_idx] - 0.5) * 2
        p_idx = rem_idx[conf > PSEUDO_CONFIDENCE] # Mẫu dễ (Pseudo)
        X_p, y_p = X_batch[p_idx], (y_prob[p_idx] > 0.5).astype(int).flatten()
        
        if len(X_p) > 0: return np.concatenate([X_q, X_p]), np.concatenate([y_q, y_p])
        else: return X_q, y_q

class SmartRehearsalBuffer:
    def __init__(self, max_size):
        self.max_size = max_size; self.buffer = [] 
    def add(self, x, y, unc):
        for i in range(len(x)): self.buffer.append((x[i], y[i], unc))
        self.buffer.sort(key=lambda x: x[2], reverse=True)
        if len(self.buffer) > self.max_size:
            n_h = int(self.max_size*0.6)
            self.buffer = self.buffer[:n_h] + random.sample(self.buffer[n_h:], self.max_size - n_h)
    def get_sample(self, n):
        if not self.buffer: return np.array([]), np.array([])
        s = random.sample(self.buffer, min(len(self.buffer), n))
        return np.array([x[0] for x in s]), np.array([x[1] for x in s])

# ==========================================
# 3. KHỞI TẠO
# ==========================================
print(">>> Đang load Model AI (Active Learning Mode)...")
try:
    model = tf.keras.models.load_model(
        MODEL_PATH, 
        custom_objects={'AttentionBlock': AttentionBlock, 'MCDropout': MCDropout}
    )
    # [QUAN TRỌNG] Compile lại để có thể train_on_batch
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-5), loss='binary_crossentropy', metrics=['accuracy'])
    print("✅ Model đã sẵn sàng để học!")
except Exception as e:
    print(f"❌ Lỗi load model: {e}"); exit()

try:
    consumer = KafkaConsumer(
        TOPIC, bootstrap_servers=['localhost:9092'],
        auto_offset_reset='latest', value_deserializer=lambda x: json.loads(x.decode('utf-8'))
    )
    print(f">>> Đang lắng nghe topic '{TOPIC}'...")
except: print("❌ Lỗi Kafka!"); exit()

if not os.path.exists(LOG_FILE):
    with open(LOG_FILE, 'w', newline='', encoding='utf-8') as f:
        csv.writer(f).writerow(["Time", "Status", "Attacks", "Type", "Accuracy", "Uncertainty", "Latency", "Action"])

# Init Active Modules
active_strategy = ActiveStrategy(budget_ratio=LABELING_BUDGET)
buffer = SmartRehearsalBuffer(REHEARSAL_SIZE)
unc_hist = deque(maxlen=UNCERTAINTY_WINDOW)

buffer_features, buffer_labels = [], []
history_acc, history_unc = [], []
drift_points = [] # Lưu vị trí drift để vẽ
batch_count = 0

print(">>> Hệ thống Adaptive NIDS đang chạy...")

# ==========================================
# 4. MAIN LOOP
# ==========================================
try:
    for message in consumer:
        data = message.value
        buffer_features.append(data['features'])
        buffer_labels.append(data['true_label'])
        
        if len(buffer_features) >= BATCH_SIZE:
            batch_count += 1
            
            # 1. Prepare Data
            X_raw = np.array(buffer_features)
            X_batch = np.expand_dims(X_raw, axis=1).repeat(10, axis=1)
            y_true = np.array(buffer_labels)
            y_true_bin = (y_true > 0).astype(int)
            
            # 2. Predict
            t0 = time.time()
            preds = model.predict(X_batch, verbose=0)
            unc = get_uncertainty(model, X_batch).numpy()
            t1 = time.time()
            latency = (t1 - t0) * 1000
            
            unc_mean = np.mean(unc)
            pred_labels = (preds > 0.5).astype(int).flatten()
            accuracy = accuracy_score(y_true_bin, pred_labels) * 100
            
            # 3. Drift Detection & Active Learning
            is_drift = False
            action = "Monitor"
            unc_hist.append(unc_mean)
            
            if len(unc_hist) == UNCERTAINTY_WINDOW:
                thresh = np.mean(unc_hist) + 2.0 * np.std(unc_hist)
                if unc_mean > thresh and unc_mean > 0.001:
                    is_drift = True
            
            if is_drift:
                action = "🔄 LEARNING"
                drift_points.append(batch_count)
                
                # A. Chọn mẫu (Active)
                X_sel, y_sel = active_strategy.select_samples(X_batch, y_true_bin, preds.flatten(), unc)
                
                # B. Lấy ký ức cũ (Replay)
                X_old, y_old = buffer.get_sample(500)
                
                # C. Học lại (Update Weights)
                if len(X_old) > 0:
                    X_train = np.concatenate([X_sel, X_old])
                    y_train = np.concatenate([y_sel, y_old])
                    model.train_on_batch(X_train, y_train)
                
                unc_hist.clear() # Reset ngưỡng
            
            # 4. Buffer Update
            buffer.add(X_batch, y_true_bin, unc_mean)
            
            # 5. Display & Log
            timestamp = datetime.now().strftime("%H:%M:%S")
            attack_indices = np.where(pred_labels == 1)[0]
            n_attacks = len(attack_indices)
            
            attack_type_str = "None"
            status_icon = "✅"
            if n_attacks > 0:
                status_icon = "⚠️"
                real_attack_labels = y_true[attack_indices]
                if len(real_attack_labels) > 0:
                    most_freq = np.argmax(np.bincount(real_attack_labels))
                    attack_type_str = CLASS_NAMES[most_freq] if most_freq < len(CLASS_NAMES) else "Unknown"
                
                print(f"[{timestamp}] {status_icon} PHÁT HIỆN: {n_attacks} ({attack_type_str}) | Acc: {accuracy:.1f}% | {action}")
            else:
                print(f"[{timestamp}] ✅ Mạng ổn định. | Acc: {accuracy:.1f}% | {action}", end='\r')
            
            # Ghi CSV
            with open(LOG_FILE, 'a', newline='', encoding='utf-8') as f:
                csv.writer(f).writerow([timestamp, "Attack" if n_attacks>0 else "Safe", n_attacks, attack_type_str, round(accuracy, 2), round(unc_mean, 4), round(latency, 1), action])
                
            history_acc.append(accuracy)
            history_unc.append(unc_mean)
            buffer_features = []
            buffer_labels = []

except KeyboardInterrupt:
    print("\n\n>>> Đang tạo biểu đồ báo cáo chi tiết...")
    
    if len(history_acc) > 0:
        # Tạo figure với 2 trục Y (Accuracy & Uncertainty)
        fig, ax1 = plt.subplots(figsize=(14, 7))
        
        # Trục X: Batch
        batches = range(len(history_acc))
        
        # --- TRỤC TRÁI: ACCURACY ---
        color_acc = 'tab:blue'
        ax1.set_xlabel('Batch Sequence', fontsize=12)
        ax1.set_ylabel('Accuracy (%)', color=color_acc, fontsize=12, fontweight='bold')
        # Vẽ đường Accuracy nét liền
        line1, = ax1.plot(batches, history_acc, color=color_acc, linewidth=2, label='Accuracy (Real-time Val)')
        ax1.tick_params(axis='y', labelcolor=color_acc)
        ax1.set_ylim(0, 105)
        ax1.grid(True, alpha=0.3)

        # --- TRỤC PHẢI: UNCERTAINTY ---
        ax2 = ax1.twinx()
        color_unc = 'tab:orange'
        ax2.set_ylabel('Uncertainty (Variance)', color=color_unc, fontsize=12, fontweight='bold')
        # Vẽ đường Uncertainty nét mờ hơn
        line2, = ax2.plot(batches, history_unc, color=color_unc, alpha=0.6, linestyle='-', label='Uncertainty')
        ax2.tick_params(axis='y', labelcolor=color_unc)
        
        # --- ĐÁNH DẤU ĐIỂM HỌC LẠI (RETRAINING) ---
        # Vẽ các vạch dọc màu đỏ tại những batch có Drift
        if len(drift_points) > 0:
            for i, d_idx in enumerate(drift_points):
                # Chỉ vẽ label cho vạch đầu tiên để đỡ rối legend
                label = 'Retraining Triggered' if i == 0 else None
                plt.axvline(x=d_idx, color='red', linestyle='--', alpha=0.5, label=label)

        # --- TIÊU ĐỀ & CHÚ THÍCH ---
        plt.title('Real-time Active Learning Performance: Accuracy vs. Uncertainty', fontsize=14, pad=20)
        
        # Gộp legend của 2 trục
        lines = [line1, line2]
        labels = [l.get_label() for l in lines]
        if len(drift_points) > 0:
            # Thêm legend cho vạch đỏ thủ công
            import matplotlib.lines as mlines
            red_line = mlines.Line2D([], [], color='red', linestyle='--', label='Retraining Triggered')
            lines.append(red_line)
            labels.append('Retraining Triggered')
            
        ax1.legend(lines, labels, loc='lower right')
        
        plt.tight_layout()
        plt.savefig(PLOT_FILE, dpi=300) # Lưu ảnh nét cao
        print(f"✅ Đã lưu biểu đồ: {os.path.abspath(PLOT_FILE)}")
    
    print("Đã dừng hệ thống.")