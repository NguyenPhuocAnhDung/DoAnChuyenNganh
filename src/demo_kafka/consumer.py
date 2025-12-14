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
import matplotlib.pyplot as plt # Thêm thư viện vẽ

# --- CẤU HÌNH ---
MODEL_PATH = "results/models/CNN_GRU_Attention.h5"
TOPIC = "nids-traffic"
BATCH_SIZE = 256
LOG_FILE = "attack_report_detail.csv"
PLOT_FILE = "demo_performance.png"

# Danh sách tên lớp (Map từ 0-7)
CLASS_NAMES = ['Normal', 'DoS', 'PortScan', 'Botnet', 'BruteForce', 'WebAttack', 'Infiltration', 'DDoS']

# --- CUSTOM LAYERS ---
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

# --- LOAD MODEL ---
print(">>> Đang load Model AI...")
try:
    model = tf.keras.models.load_model(
        MODEL_PATH, 
        custom_objects={'AttentionBlock': AttentionBlock, 'MCDropout': MCDropout}
    )
    print("✅ Model đã sẵn sàng!")
except Exception as e:
    print(f"❌ Lỗi load model: {e}")
    exit()

# --- KAFKA CONSUMER ---
try:
    consumer = KafkaConsumer(
        TOPIC,
        bootstrap_servers=['localhost:9092'],
        auto_offset_reset='latest',
        value_deserializer=lambda x: json.loads(x.decode('utf-8'))
    )
    print(f">>> Đang lắng nghe topic '{TOPIC}'...")
except Exception as e:
    print("❌ Lỗi kết nối Kafka.")
    exit()

# Khởi tạo file log
if not os.path.exists(LOG_FILE):
    with open(LOG_FILE, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(["Time", "Status", "Attacks", "Type", "Accuracy", "Latency"])

buffer_features = []
buffer_labels = []

# Biến lưu lịch sử để vẽ biểu đồ
history_acc = []
history_lat = []

print(">>> Hệ thống giám sát chi tiết đang chạy (Ctrl+C để dừng và vẽ biểu đồ)...")

try:
    for message in consumer:
        data = message.value
        buffer_features.append(data['features'])
        buffer_labels.append(data['true_label'])
        
        if len(buffer_features) >= BATCH_SIZE:
            # 1. Xử lý dữ liệu
            X_batch_raw = np.array(buffer_features)
            X_batch = np.expand_dims(X_batch_raw, axis=1).repeat(10, axis=1)
            y_true = np.array(buffer_labels)
            
            # 2. Dự đoán
            t0 = time.time()
            preds = model.predict(X_batch, verbose=0)
            t1 = time.time()
            latency = (t1 - t0) * 1000
            
            pred_labels = (preds > 0.5).astype(int).flatten()
            
            # 3. Phân tích kết quả
            # Chuyển đổi label thật (0-7) sang nhị phân (0: Normal, 1-7: Attack)
            y_true_bin = (y_true > 0).astype(int)
            accuracy = accuracy_score(y_true_bin, pred_labels) * 100
            
            # Lưu history
            history_acc.append(accuracy)
            history_lat.append(latency)
            
            # Đếm số lượng tấn công
            attack_indices = np.where(pred_labels == 1)[0]
            n_attacks = len(attack_indices)
            
            timestamp = datetime.now().strftime("%H:%M:%S")
            status = "AN TOÀN"
            attack_type_str = "None"
            
            if n_attacks > 0:
                status = "CẢNH BÁO"
                # Lấy loại tấn công phổ biến nhất từ nhãn thật (để đối chiếu)
                real_attack_labels = y_true[attack_indices]
                if len(real_attack_labels) > 0:
                    counts = np.bincount(real_attack_labels)
                    most_freq_type = np.argmax(counts)
                    if most_freq_type < len(CLASS_NAMES):
                        attack_type_str = CLASS_NAMES[most_freq_type]
                    else:
                        attack_type_str = f"Unknown({most_freq_type})"
                
                print(f"[{timestamp}] ⚠️  PHÁT HIỆN: {n_attacks} gói tin | Loại: {attack_type_str} | Acc: {accuracy:.1f}% | Lat: {latency:.1f}ms")
            else:
                print(f"[{timestamp}] ✅ Mạng ổn định. | Acc: {accuracy:.1f}% | Lat: {latency:.1f}ms", end='\r')
            
            # Ghi log
            with open(LOG_FILE, mode='a', newline='', encoding='utf-8') as file:
                writer = csv.writer(file)
                writer.writerow([timestamp, status, n_attacks, attack_type_str, round(accuracy, 2), round(latency, 2)])
                
            buffer_features = []
            buffer_labels = []

except KeyboardInterrupt:
    print("\n\n>>> Đang tạo biểu đồ báo cáo...")
    
    # Vẽ biểu đồ khi dừng chương trình
    if len(history_acc) > 0:
        plt.figure(figsize=(12, 5))
        
        # Biểu đồ Accuracy
        plt.subplot(1, 2, 1)
        plt.plot(history_acc, label='Accuracy (%)', color='blue')
        plt.title('Real-time Accuracy')
        plt.xlabel('Batch')
        plt.ylabel('Accuracy')
        plt.ylim(0, 105)
        plt.grid(True)
        
        # Biểu đồ Latency
        plt.subplot(1, 2, 2)
        plt.plot(history_lat, label='Latency (ms)', color='orange')
        plt.title('Real-time Latency')
        plt.xlabel('Batch')
        plt.ylabel('ms')
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig(PLOT_FILE)
        print(f"✅ Đã lưu biểu đồ tại: {os.path.abspath(PLOT_FILE)}")
    
    print("Đã dừng hệ thống.")