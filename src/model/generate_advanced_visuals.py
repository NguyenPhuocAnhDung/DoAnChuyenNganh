import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
import os
from sklearn.manifold import TSNE
from sklearn.metrics import confusion_matrix

# Cấu hình
plt.rcParams.update({'font.size': 12, 'font.family': 'serif'})
sns.set_style("whitegrid")

# --- ĐƯỜNG DẪN (Giữ nguyên như cũ) ---
BASE_DIR = "../../baocao"
DATA_PATH = "../../dataset/processed/processedstreamvs2.4/processed_online_stream.parquet"
MODEL_PATH = os.path.join(BASE_DIR, "main_active_learning/models/final_model_A.h5")
LOG_CD_AHAL = os.path.join(BASE_DIR, "main_active_learning/plots/history_active_learning.csv")
LOG_WEAK_AL = os.path.join(BASE_DIR, "main_cnn_attention_ative_learning/plots/history_cnn_attention_AL.csv")

OUTPUT_DIR = "./final_paper_images_real/"
if not os.path.exists(OUTPUT_DIR): os.makedirs(OUTPUT_DIR)

# --- 1. ĐỊNH NGHĨA CUSTOM LAYERS ĐỂ LOAD MODEL ---
# [FIX] Thêm decorator register_keras_serializable cho chắc chắn
@tf.keras.utils.register_keras_serializable()
class MCDropout(tf.keras.layers.Dropout):
    def call(self, inputs, training=None): return super().call(inputs, training=True)

@tf.keras.utils.register_keras_serializable()
class AttentionBlock(tf.keras.layers.Layer):
    def __init__(self, **kwargs): super(AttentionBlock, self).__init__(**kwargs)
    def build(self, input_shape):
        self.W = self.add_weight(name='att_weight', shape=(input_shape[-1], 1), initializer='normal')
        self.b = self.add_weight(name='att_bias', shape=(input_shape[1], 1), initializer='zeros')
        super(AttentionBlock, self).build(input_shape)
    def call(self, x):
        e = tf.keras.backend.tanh(tf.keras.backend.dot(x, self.W) + self.b)
        a = tf.keras.backend.softmax(e, axis=1)
        output = x * a
        return tf.keras.backend.sum(output, axis=1)

# [QUAN TRỌNG] SỬA LỖI UNKNOWN LAYER TẠI ĐÂY
custom_objects = {
    'MCDropout': MCDropout, 
    'Custom>MCDropout': MCDropout,          # <--- DÒNG FIX LỖI CHÍNH
    'AttentionBlock': AttentionBlock,
    'Custom>AttentionBlock': AttentionBlock, # <--- THÊM DỰ PHÒNG
    'F1Score': None, 
    'SparseFocalLoss': None
}

# ==========================================
# A. VẼ HÌNH 8: t-SNE TỪ DỮ LIỆU & MODEL THẬT
# ==========================================
def generate_real_tsne():
    print("⏳ [Fig 8] Đang load model và dữ liệu để vẽ t-SNE (Mất khoảng 30s)...")
    try:
        # Load Model với custom_objects đã sửa
        model = tf.keras.models.load_model(MODEL_PATH, custom_objects=custom_objects, compile=False)
        
        # Tạo mô hình trích xuất đặc trưng (Lấy output lớp sát cuối)
        # Kiểm tra tên lớp để lấy đúng (thường là flatten hoặc dense gần cuối)
        # Ở đây ta lấy layer thứ -2 (áp chót)
        feature_extractor = tf.keras.Model(inputs=model.inputs, outputs=model.layers[-2].output)

        # 2. Load Dữ liệu (Lấy 1000 mẫu để vẽ cho nhẹ)
        df = pd.read_parquet(DATA_PATH)
        
        # Lấy mẫu đại diện: 500 mẫu đầu (Normal) + 500 mẫu đoạn Drift (Attack)
        df_normal = df.iloc[:500]
        # Lấy đoạn drift ở khoảng batch 152 (152 * 2560 ~ dòng 389000). 
        # Nếu file nhỏ hơn thì lấy 500 dòng cuối.
        idx_drift = 38000 if len(df) > 40000 else len(df) - 500
        df_attack = df.iloc[idx_drift : idx_drift+500]
        
        df_sample = pd.concat([df_normal, df_attack])
        X_sample = df_sample.drop(columns=['Label', 'Label_Multi'], errors='ignore').values
        y_sample = df_sample['Label'].values
        
        # Reshape cho CNN-GRU (samples, 10, features)
        TIME_STEPS = 10
        n_feats = X_sample.shape[1]
        limit = (len(X_sample) // TIME_STEPS) * TIME_STEPS
        X_seq = X_sample[:limit].reshape(-1, TIME_STEPS, n_feats)
        y_seq = y_sample[:limit:TIME_STEPS]

        # 3. Trích xuất đặc trưng
        features = feature_extractor.predict(X_seq, verbose=0)
        
        # 4. Chạy t-SNE
        tsne = TSNE(n_components=2, random_state=42, perplexity=30)
        X_tsne = tsne.fit_transform(features)
        
        # 5. Vẽ
        plt.figure(figsize=(10, 7))
        scatter = plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=y_seq, cmap='coolwarm', alpha=0.6, s=50)
        cbar = plt.colorbar(scatter)
        cbar.set_label('Label (0: Normal, 1: Attack)')
        
        plt.title('Real t-SNE Visualization: Feature Space Separation')
        plt.xlabel('t-SNE Dim 1')
        plt.ylabel('t-SNE Dim 2')
        
        # Chú thích
        plt.text(X_tsne[0,0], X_tsne[0,1], "Normal Domain", fontsize=12, fontweight='bold', color='blue')
        plt.text(X_tsne[-1,0], X_tsne[-1,1], "Attack/Drift Domain", fontsize=12, fontweight='bold', color='red')

        save_p = os.path.join(OUTPUT_DIR, "Fig8_Real_tSNE.png")
        plt.savefig(save_p, dpi=300)
        print(f"✅ Đã tạo Fig 8: {save_p}")
        
    except Exception as e:
        print(f"❌ Vẫn lỗi t-SNE: {e}")

# ==========================================
# B. VẼ HÌNH 6b: SO SÁNH AL (TỪ LOG)
# ==========================================
def generate_real_al_comparison():
    print("⏳ [Fig 6b] Đang so sánh hiệu năng Active Learning từ log...")
    try:
        if os.path.exists(LOG_CD_AHAL) and os.path.exists(LOG_WEAK_AL):
            df_strong = pd.read_csv(LOG_CD_AHAL)
            df_weak = pd.read_csv(LOG_WEAK_AL)
            
            df_strong.columns = [c.lower() for c in df_strong.columns]
            df_weak.columns = [c.lower() for c in df_weak.columns]
            
            plt.figure(figsize=(10, 6))
            
            win = 10 # Tăng window để đường mượt hơn
            plt.plot(df_strong['accuracy'].rolling(win).mean()*100, label='CD-AHAL (Strong Arch + AL)', color='blue', linewidth=2)
            plt.plot(df_weak['accuracy'].rolling(win).mean()*100, label='Weak AL (CNN-Attn + AL)', color='orange', linestyle='--')
            
            plt.title('Recovery Performance Comparison (Real Data)')
            plt.xlabel('Streaming Batches')
            plt.ylabel('Accuracy (%)')
            plt.legend()
            plt.grid(True, linestyle='--', alpha=0.6)
            
            save_p = os.path.join(OUTPUT_DIR, "Fig6b_Real_AL_Comparison.png")
            plt.savefig(save_p, dpi=300)
            print(f"✅ Đã tạo Fig 6b: {save_p}")
        else:
            print("⚠️ Thiếu file log CSV.")
    except Exception as e:
        print(f"❌ Lỗi vẽ Fig 6b: {e}")

# ==========================================
# C. VẼ HÌNH 4b: MA TRẬN NHẦM LẪN (TỪ MODEL THẬT)
# ==========================================
def generate_real_cm():
    print("⏳ [Fig 4b] Đang tạo Confusion Matrix từ Model thật...")
    try:
        model = tf.keras.models.load_model(MODEL_PATH, custom_objects=custom_objects, compile=False)
        
        # Load 2000 mẫu cuối cùng của luồng dữ liệu
        df = pd.read_parquet(DATA_PATH)
        df_test = df.iloc[-2000:] 
        
        X_test = df_test.drop(columns=['Label', 'Label_Multi'], errors='ignore').values
        y_test = df_test['Label'].values
        
        TIME_STEPS = 10
        limit = (len(X_test) // TIME_STEPS) * TIME_STEPS
        X_seq = X_test[:limit].reshape(-1, TIME_STEPS, X_test.shape[1])
        y_seq = y_test[:limit:TIME_STEPS]
        
        y_prob = model.predict(X_seq, verbose=0)
        y_pred = (y_prob > 0.5).astype(int).flatten()
        
        cm = confusion_matrix(y_seq, y_pred)
        plt.figure(figsize=(7, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=['Normal', 'Attack'], yticklabels=['Normal', 'Attack'])
        plt.title('Real Confusion Matrix (Phase 3 Snapshot)')
        plt.ylabel('True Label'); plt.xlabel('Predicted Label')
        
        save_p = os.path.join(OUTPUT_DIR, "Fig4b_Real_CM.png")
        plt.savefig(save_p, dpi=300)
        print(f"✅ Đã tạo Fig 4b: {save_p}")
        
    except Exception as e:
        print(f"❌ Lỗi vẽ Fig 4b: {e}")

if __name__ == "__main__":
    generate_real_tsne()
    generate_real_al_comparison()
    generate_real_cm()
    print(f"\n🎉 HOÀN TẤT! Ảnh thật được lưu tại: {OUTPUT_DIR}")