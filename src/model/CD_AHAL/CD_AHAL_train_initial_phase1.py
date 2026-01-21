import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model, backend as K
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix, classification_report
from tabulate import tabulate

# ==========================================
# 1. CẤU HÌNH & PATH
# ==========================================
DATA_PATH = "D:/DoAnChuyenNganh/dataset/processed_v3_unified"
BASE_OUTPUT = "../../../baocao/CD_AHAL_FINAL_FULL_METRICS_FINAL"
MODEL_PATH = os.path.join(BASE_OUTPUT, "models")
PLOT_PATH = os.path.join(BASE_OUTPUT, "plots_train")
REPORT_PATH = os.path.join(BASE_OUTPUT, "reports_train")

# Tạo thư mục nếu chưa có
for p in [MODEL_PATH, PLOT_PATH, REPORT_PATH]: 
    if not os.path.exists(p): os.makedirs(p)

INITIAL_TRAIN_FILE = os.path.join(DATA_PATH, "processed_initial_train_balanced.parquet")

TIME_STEPS = 10
N_CLASSES_MULTI = 8
DROPOUT_RATE = 0.3
EPOCHS = 100
BATCH_SIZE = 512
CLASS_NAMES = ['Normal', 'DoS', 'PortScan', 'Botnet', 'BruteForce', 'WebAttack', 'Infiltration', 'DDoS']

# ==========================================
# 2. CUSTOM LAYERS & METRICS
# ==========================================
@tf.keras.utils.register_keras_serializable()
class AttentionLayer(layers.Layer):
    def __init__(self, **kwargs): super().__init__(**kwargs)
    def build(self, input_shape):
        self.W = self.add_weight(name='w', shape=(input_shape[-1], 1), initializer='normal')
        self.b = self.add_weight(name='b', shape=(input_shape[1], 1), initializer='zeros')
        super().build(input_shape)
    def call(self, x):
        e = K.tanh(K.dot(x, self.W) + self.b)
        a = K.softmax(e, axis=1)
        return K.sum(x * a, axis=1)

@tf.keras.utils.register_keras_serializable()
class MCDropout(layers.Dropout):
    def call(self, inputs, training=None): return super().call(inputs, training=True)

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
        return 2 * ((p * r) / (p + r + tf.keras.backend.epsilon()))
    def reset_state(self):
        self.precision.reset_state()
        self.recall.reset_state()

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
# 3. UTILS
# ==========================================
def prepare_sequences(X, y, time_steps):
    if len(X) < time_steps: return np.array([]), np.array([])
    n = len(X) // time_steps
    X_seq = X[:n*time_steps].reshape((n, time_steps, X.shape[1]))
    y_seq = y[time_steps-1::time_steps]
    return X_seq, y_seq

def plot_history(history, filename):
    # Vẽ biểu đồ Loss và Accuracy
    metrics = ['loss', 'accuracy']
    plt.figure(figsize=(12, 5))
    for i, m in enumerate(metrics):
        if m in history.history:
            plt.subplot(1, 2, i+1)
            plt.plot(history.history[m], label=f'Train {m}')
            plt.plot(history.history[f'val_{m}'], label=f'Val {m}')
            plt.title(m.upper())
            plt.legend()
            plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_PATH, filename), dpi=300)
    plt.close()

def plot_cm(y_true, y_pred, classes, title, filename, cmap='Blues'):
    # Vẽ ma trận nhầm lẫn
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap=cmap, xticklabels=classes, yticklabels=classes)
    plt.title(title)
    plt.ylabel('True')
    plt.xlabel('Predicted')
    plt.savefig(os.path.join(PLOT_PATH, filename), bbox_inches='tight', dpi=300)
    plt.close()

# ==========================================
# 4. MODELS
# ==========================================
def create_binary_model(input_shape):
    inputs = layers.Input(shape=input_shape)
    x = layers.Conv1D(128, 3, activation='relu', padding='same')(inputs)
    x = layers.MaxPooling1D(2)(x)
    x = MCDropout(DROPOUT_RATE)(x)
    x = layers.Bidirectional(layers.GRU(128, return_sequences=True))(x)
    x = AttentionLayer()(x)
    outputs = layers.Dense(1, activation='sigmoid')(x)
    
    model = Model(inputs, outputs, name="Binary")
    model.compile('adam', 'binary_crossentropy', 
                  metrics=['accuracy', tf.keras.metrics.Precision(name='precision'), 
                           tf.keras.metrics.Recall(name='recall'), F1Score(name='f1_score')])
    return model

def create_multiclass_model(input_shape):
    inputs = layers.Input(shape=input_shape)
    x = layers.Conv1D(64, 3, activation='relu', padding='same')(inputs)
    x = layers.MaxPooling1D(2)(x)
    x = MCDropout(DROPOUT_RATE)(x)
    x = layers.Bidirectional(layers.GRU(64, return_sequences=True))(x)
    x = AttentionLayer()(x)
    outputs = layers.Dense(N_CLASSES_MULTI, activation='softmax')(x)
    
    model = Model(inputs, outputs, name="Multi")
    model.compile('adam', SparseFocalLoss(), metrics=['accuracy'])
    return model

# ==========================================
# 5. MAIN
# ==========================================
def main():
    print("--- PHASE 1: TRAIN INITIAL MODELS ---")
    if not os.path.exists(INITIAL_TRAIN_FILE):
        print(f"❌ File not found: {INITIAL_TRAIN_FILE}"); return

    df = pd.read_parquet(INITIAL_TRAIN_FILE)
    feat_cols = [c for c in df.columns if c not in ['Label', 'Label_Multi', 'Label_Bin']]
    
    # Chia dữ liệu Train/Val/Test
    df_train, df_temp = train_test_split(df, test_size=0.3, random_state=42, stratify=df['Label'])
    df_val, df_test = train_test_split(df_temp, test_size=0.5, random_state=42, stratify=df_temp['Label'])
    
    def get_d(d):
        X, yb = prepare_sequences(d[feat_cols].values, d['Label'].values, TIME_STEPS)
        _, ym = prepare_sequences(d[feat_cols].values, d['Label_Multi'].values, TIME_STEPS)
        return X, yb, ym

    X_train, yb_train, ym_train = get_d(df_train)
    X_val, yb_val, ym_val = get_d(df_val)
    X_test, yb_test, ym_test = get_d(df_test)
    
    print(f"Train data shape: {X_train.shape}")

    # ------------------------------------------
    # 1. Train Binary Model
    # ------------------------------------------
    print("\n[1] Training Binary Model A...")
    model_A = create_binary_model(X_train.shape[1:])
    hist_A = model_A.fit(X_train, yb_train, validation_data=(X_val, yb_val), epochs=EPOCHS, batch_size=BATCH_SIZE, 
                         callbacks=[EarlyStopping(patience=5, restore_best_weights=True)], verbose=1)
    
    model_A.save(os.path.join(MODEL_PATH, "CD_AHAL_model_A_initial.h5"))
    plot_history(hist_A, "History_Binary.png") # Xuất biểu đồ train

    # ------------------------------------------
    # 2. Train Multiclass Model
    # ------------------------------------------
    print("\n[2] Training Multiclass Model B...")
    idx_atk = np.where(ym_train > 0)[0]; idx_val = np.where(ym_val > 0)[0]
    
    model_B = create_multiclass_model(X_train.shape[1:])
    hist_B = model_B.fit(X_train[idx_atk], ym_train[idx_atk], validation_data=(X_val[idx_val], ym_val[idx_val]),
                         epochs=EPOCHS, batch_size=BATCH_SIZE, callbacks=[EarlyStopping(patience=5)], verbose=1)
    
    model_B.save(os.path.join(MODEL_PATH, "CD_AHAL_model_B_initial.h5"))
    plot_history(hist_B, "History_Multiclass.png") # Xuất biểu đồ train

    # ------------------------------------------
    # 3. TEST EVALUATION & PLOTS
    # ------------------------------------------
    print("\n--- FINAL TEST EVALUATION ---")
    
    # --- Đánh giá Binary ---
    print("\n[Binary Metrics]")
    results_bin = model_A.evaluate(X_test, yb_test, return_dict=True, verbose=0)
    print(tabulate(pd.DataFrame([results_bin]), headers='keys', tablefmt='grid'))
    
    # Dự đoán & Vẽ CM Binary
    pb = (model_A.predict(X_test, verbose=0) > 0.5).astype(int).flatten()
    plot_cm(yb_test, pb, ['Normal', 'Attack'], "Binary Test CM", "CM_Binary_Test.png")
    
    # --- Đánh giá Multiclass ---
    print("\n[Multiclass Metrics]")
    # Chỉ dự đoán lớp cụ thể cho những mẫu được dự đoán là Attack
    pm = np.zeros_like(ym_test)
    idx_pred_attack = np.where(pb == 1)[0]
    
    if len(idx_pred_attack) > 0:
        pred_probs = model_B.predict(X_test[idx_pred_attack], verbose=0)
        pm[idx_pred_attack] = np.argmax(pred_probs, axis=1)
    
    print(classification_report(ym_test, pm, target_names=CLASS_NAMES, zero_division=0))
    
    # Vẽ CM Multiclass
    plot_cm(ym_test, pm, CLASS_NAMES, "Multi Test CM", "CM_Multiclass_Test.png", cmap="Reds")
    
    print(f"✅ Training Done. Models & Plots saved to {BASE_OUTPUT}")

if __name__ == "__main__":
    main()