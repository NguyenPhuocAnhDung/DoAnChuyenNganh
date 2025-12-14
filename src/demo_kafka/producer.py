import time
import json
import pandas as pd
from kafka import KafkaProducer
import os

# --- CẤU HÌNH ĐƯỜNG DẪN (CHẠY TỪ D:\DACN) ---
# Đã sửa: Bỏ ../.. và trỏ đúng file parquet
DATA_PATH = "D:/DACN/dataset/processed/processedstreamvs2.4/processed_online_stream.parquet"
TOPIC = "nids-traffic"

def json_serializer(data):
    return json.dumps(data).encode("utf-8")

try:
    producer = KafkaProducer(
        bootstrap_servers=['localhost:9092'],
        value_serializer=json_serializer
    )
except:
    print("❌ Lỗi kết nối Kafka. Hãy chạy 'docker compose up' trước!")
    exit()

print(f">>> Đang tải dữ liệu từ: {DATA_PATH}")
if not os.path.exists(DATA_PATH):
    print(f"❌ Không tìm thấy file dữ liệu tại: {os.path.abspath(DATA_PATH)}")
    exit()

df = pd.read_parquet(DATA_PATH)
feat_cols = [c for c in df.columns if c not in ['Label', 'Label_Multi']]

print(">>> BẮT ĐẦU BẮN DỮ LIỆU (Giả lập mạng)...")

count = 0
for index, row in df.iterrows():
    message = {
        "timestamp": time.time(),
        "features": row[feat_cols].values.tolist(), 
        "true_label": int(row['Label'])
    }
    
    producer.send(TOPIC, message)
    count += 1
    
    if count % 256 == 0: # In log mỗi batch
        print(f"[Producer] Đã gửi {count} gói tin...", end='\r')
        time.sleep(0.05) # Tốc độ bắn tin

print("\n✅ Đã gửi hết dữ liệu!")

# [FIX] Đóng producer đàng hoàng để tránh lỗi Timeout
try:
    producer.flush() # Ép gửi hết tin còn tồn đọng
    producer.close() # Đóng kết nối an toàn
    print("🔌 Đã đóng kết nối Kafka.")
except Exception as e:
    print(f"⚠️ Lỗi khi đóng kết nối: {e}")