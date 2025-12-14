import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model 
from tensorflow.keras.callbacks import EarlyStopping 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
# [UPDATE] Import thêm các metrics cần thiết
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
from sklearn.utils import class_weight 
from collections import deque
import random
import os
import time

# Tắt log rác
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# ==========================================
# 1. CẤU HÌNH HỆ THỐNG
# ==========================================
DATA_PATH = "D:/DACN/dataset/processed/processedstreamvs2.4"
PLOT_PATH = "../../baocao/plots" 
MODEL_PATH = "../../baocao/models"

if not os.path.exists(PLOT_PATH): os.makedirs(PLOT_PATH)
if not os.path.exists(MODEL_PATH): os.makedirs(MODEL_PATH)

INITIAL_TRAIN_FILE = os.path.join(DATA_PATH, "processed_initial_train_balanced.parquet")
ONLINE_STREAM_FILE = os.path.join(DATA_PATH, "processed_online_stream.parquet")

# HYPERPARAMETERS
TIME_STEPS = 10 
BATCH_SIZE = 256 
RAW_BATCH_SIZE = BATCH_SIZE * TIME_STEPS 
DROPOUT_RATE = 0.3
MC_SAMPLES = 10 
ONLINE_EPOCHS = 15 

# GAN CONFIG
GAN_LATENT_DIM = 100  
GAN_EPOCHS = 5        

# DRIFT CONFIG
UNCERTAINTY_WINDOW = 50      
NORMAL_CLASS_BOOST = 3.0      

CLASS_NAMES = ['Normal', 'DoS', 'PortScan', 'Botnet', 'BruteForce', 'WebAttack', 'Infiltration', 'DDoS']

# ==========================================
# 2. MODULE GENERATIVE REPLAY (C-GAN)
# ==========================================
class ConditionalGAN:
    def __init__(self, n_features, n_classes=2):
        self.n_features = n_features
        self.n_classes = n_classes
        self.latent_dim = GAN_LATENT_DIM
        
        self.generator = self.build_generator()
        self.discriminator = self.build_discriminator()
        
        self.g_optimizer = keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.5)
        self.d_optimizer = keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.5)
        self.loss_fn = keras.losses.BinaryCrossentropy()

    def build_generator(self):
        noise = layers.Input(shape=(self.latent_dim,))
        label = layers.Input(shape=(1,))
        label_embedding = layers.Flatten()(layers.Embedding(self.n_classes, self.latent_dim)(label))
        model_input = layers.multiply([noise, label_embedding])
        
        x = layers.Dense(128, activation='relu')(model_input)
        x = layers.BatchNormalization()
        x = layers.Dense(256, activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dense(self.n_features, activation='linear')(x)
        return Model([noise, label], x, name="Generator")

    def build_discriminator(self):
        features = layers.Input(shape=(self.n_features,))
        label = layers.Input(shape=(1,))
        label_embedding = layers.Flatten()(layers.Embedding(self.n_classes, self.n_features)(label))
        model_input = layers.multiply([features, label_embedding])
        
        x = layers.Dense(256, activation='relu')(model_input)
        x = layers.Dropout(0.3)(x)
        x = layers.Dense(128, activation='relu')(x)
        x = layers.Dropout(0.3)(x)
        validity = layers.Dense(1, activation='sigmoid')(x)
        return Model([features, label], validity, name="Discriminator")

    @tf.function
    def train_step(self, real_features, real_labels):
        batch_size = tf.shape(real_features)[0]
        noise = tf.random.normal((batch_size, self.latent_dim))
        fake_features = self.generator([noise, real_labels], training=True)
        
        with tf.GradientTape() as tape:
            d_pred_real = self.discriminator([real_features, real_labels], training=True)
            d_pred_fake = self.discriminator([fake_features, real_labels], training=True)
            d_loss = 0.5 * (self.loss_fn(tf.ones_like(d_pred_real), d_pred_real) + \
                            self.loss_fn(tf.zeros_like(d_pred_fake), d_pred_fake))
            
        grads = tape.gradient(d_loss, self.discriminator.trainable_variables)
        self.d_optimizer.apply_gradients(zip(grads, self.discriminator.trainable_variables))

        with tf.GradientTape() as tape:
            fake_features = self.generator([noise, real_labels], training=True)
            d_pred_fake = self.discriminator([fake_features, real_labels], training=True)
            g_loss = self.loss_fn(tf.ones_like(d_pred_fake), d_pred_fake)
            
        grads = tape.gradient(g_loss, self.generator.trainable_variables)
        self.g_optimizer.apply_gradients(zip(grads, self.generator.trainable_variables))
        return d_loss, g_loss

    def generate_data(self, n_samples):
        # Sinh cân bằng 50-50
        labels_0 = np.zeros((n_samples // 2, 1))
        labels_1 = np.ones((n_samples - n_samples // 2, 1))
        labels_np = np.vstack([labels_0, labels_1])
        np.random.shuffle(labels_np)
        
        labels_tf = tf.convert_to_tensor(labels_np, dtype=tf.float32)
        noise = tf.random.normal((n_samples, self.latent_dim))
        generated_features = self.generator([noise, labels_tf], training=False)
        return generated_features.numpy(), labels_np.flatten()

# --- CONTROLLER ---
class DynamicController:
    def __init__(self, base_lr=1e-5, base_thresh=3.0):
        self.base_lr = base_lr; self.base_thresh = base_thresh; self.ema_uncertainty = 0.01 
    def update(self, current_uncertainty):
        self.ema_uncertainty = 0.9 * self.ema_uncertainty + 0.1 * current_uncertainty
        ratio = current_uncertainty / (self.ema_uncertainty + 1e-9)
        new_lr = min(self.base_lr * max(1.0, ratio * 20.0), 0.001) 
        new_thresh = max(1.5, self.base_thresh / max(1.0, np.log1p(ratio)))
        return new_lr, new_thresh

# ==========================================
# 3. MODEL UTIL
# ==========================================
def preprocess_batch(X): return X 

def prepare_sequences(X_flat, y_flat, time_steps):
    if len(X_flat) < time_steps: return np.array([]), np.array([])
    n = len(X_flat) // time_steps
    X_seq = X_flat[:n*time_steps].reshape((n, time_steps, X_flat.shape[1]))
    y_seq = y_flat[time_steps-1::time_steps]
    return X_seq, y_seq

def create_binary_model(input_shape):
    inputs = layers.Input(shape=input_shape)
    class MCDropout(layers.Dropout):
        def call(self, inputs): return super().call(inputs, training=True)
    
    x = layers.Conv1D(128, 3, activation='relu', padding='same')(inputs)
    x = layers.BatchNormalization(); x = layers.MaxPooling1D(2)(x); x = MCDropout(DROPOUT_RATE)(x)
    x = layers.Bidirectional(layers.GRU(64, return_sequences=False))(x)
    outputs = layers.Dense(1, activation='sigmoid')(x)
    
    model = Model(inputs, outputs)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

@tf.function(reduce_retracing=True)
def get_uncertainty(model, x_batch):
    preds = tf.stack([model(x_batch, training=False) for _ in range(MC_SAMPLES)], axis=0)
    return tf.reduce_mean(tf.math.reduce_variance(preds, axis=0))

# ==========================================
# 4. MAIN EXECUTION
# ==========================================
def main():
    print(">>> STARTING GAN REPLAY SYSTEM (FULL METRICS & PLOTS) <<<")
    
    # 1. LOAD TRAIN
    df_init = pd.read_parquet(INITIAL_TRAIN_FILE)
    feat_cols = [c for c in df_init.columns if c not in ['Label', 'Label_Multi']]
    n_features = len(feat_cols)
    X_init = df_init[feat_cols].values; y_init_bin = df_init['Label'].values
    X_seq, y_bin_seq = prepare_sequences(X_init, y_init_bin, TIME_STEPS)

    # 2. PRE-TRAIN GAN
    print("\n--- Pre-training GAN ---")
    gan_model = ConditionalGAN(n_features=n_features)
    gan_idx = np.random.choice(len(X_init), size=20000, replace=False)
    X_gan = tf.convert_to_tensor(X_init[gan_idx], dtype=tf.float32)
    y_gan = tf.convert_to_tensor(y_init_bin[gan_idx].reshape(-1, 1), dtype=tf.float32)
    
    for epoch in range(5):
        ds = tf.data.Dataset.from_tensor_slices((X_gan, y_gan)).batch(128)
        for x_b, y_b in ds: gan_model.train_step(x_b, y_b)
        print(f"   GAN Epoch {epoch+1}")

    # 3. TRAIN IDS
    print("\n--- Training IDS ---")
    early = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
    model_A = create_binary_model((TIME_STEPS, n_features))
    cw_A = class_weight.compute_class_weight('balanced', classes=np.unique(y_bin_seq), y=y_bin_seq)
    model_A.fit(X_seq, y_bin_seq, epochs=15, batch_size=256, validation_split=0.1, callbacks=[early], class_weight=dict(enumerate(cw_A)), verbose=1)

    # 4. STREAM
    print(f"\n>>> ONLINE STREAMING <<<")
    df_stream = pd.read_parquet(ONLINE_STREAM_FILE)
    X_st_raw = df_stream[feat_cols].values; y_st_bin = df_stream['Label'].values
    
    unc_hist = deque(maxlen=UNCERTAINTY_WINDOW)
    # [UPDATE] Thêm các list lưu metrics
    metrics = {'acc': [], 'unc': [], 'lat': [], 'f1': [], 'prec': [], 'rec': []}
    drift_points = []
    # [UPDATE] Lưu trữ để vẽ Confusion Matrix
    y_true_all, y_pred_all = [], []
    
    controller = DynamicController(base_lr=1e-5, base_thresh=3.0)
    opt_A = keras.optimizers.Adam(learning_rate=1e-5)
    
    # Hybrid Buffer
    real_buffer = deque(maxlen=1000)
    def add_to_real_buffer(X, y):
        for i in range(len(X)): real_buffer.append((X[i], y[i]))
    add_to_real_buffer(X_seq[:1000], y_bin_seq[:1000])

    n_batches = len(X_st_raw) // RAW_BATCH_SIZE
    
    for i in range(n_batches):
        start = i * RAW_BATCH_SIZE; end = (i+1) * RAW_BATCH_SIZE
        X_curr = X_st_raw[start:end]; y_curr_raw = y_st_bin[start:end]
        X_sq, y_bin = prepare_sequences(X_curr, y_curr_raw, TIME_STEPS)
        if len(X_sq) == 0: continue
        
        # ONLINE GAN UPDATE (Chỉ update nếu batch có đủ 2 class để tránh lỗi)
        if len(np.unique(y_curr_raw)) > 1:
            try:
                gan_model.train_step(
                    tf.convert_to_tensor(X_curr, dtype=tf.float32), 
                    tf.convert_to_tensor(y_curr_raw.reshape(-1, 1), dtype=tf.float32)
                )
            except: pass

        # INFERENCE & LATENCY
        t0 = time.time(); p_A = model_A.predict(X_sq, verbose=0); t1 = time.time()
        metrics['lat'].append((t1-t0)*1000)
        
        pred_bin = (p_A > 0.5).astype(int).flatten()
        
        # [UPDATE] Tính Full Metrics
        acc = accuracy_score(y_bin, pred_bin)
        f1 = f1_score(y_bin, pred_bin, zero_division=0)
        prec = precision_score(y_bin, pred_bin, zero_division=0)
        rec = recall_score(y_bin, pred_bin, zero_division=0)
        unc = get_uncertainty(model_A, X_sq).numpy()
        
        metrics['acc'].append(acc); metrics['unc'].append(unc)
        metrics['f1'].append(f1); metrics['prec'].append(prec); metrics['rec'].append(rec)
        
        # Lưu cho CM
        y_true_all.extend(y_bin); y_pred_all.extend(pred_bin)
        
        # UPDATE PARAMS
        dyn_lr, dyn_thresh = controller.update(unc)
        opt_A.learning_rate.assign(dyn_lr)
        
        # DRIFT CHECK
        is_drift = False
        if len(unc_hist) == UNCERTAINTY_WINDOW:
            if unc > np.mean(unc_hist) + dyn_thresh * np.std(unc_hist) and unc > 0.001: is_drift = True
        unc_hist.append(unc)
        
        if is_drift:
            drift_points.append(i)
            print(f"[DRIFT] Batch {i} | Unc: {unc:.4f} | Hybrid Replay...")
            
            # 1. Fake Data
            gen_feat, gen_lbl = gan_model.generate_data(2000)
            gen_X_seq = np.expand_dims(gen_feat, axis=1).repeat(TIME_STEPS, axis=1)
            
            # 2. Real Data
            X_real, y_real = np.array([]), np.array([])
            if len(real_buffer) > 0:
                n_take = min(len(real_buffer), 500)
                real_samples = random.sample(real_buffer, n_take)
                X_real = np.array([s[0] for s in real_samples])
                y_real = np.array([s[1] for s in real_samples])
            
            # 3. Mix
            if len(X_real) > 0:
                X_ft = np.concatenate((X_sq, gen_X_seq, X_real))
                y_ft = np.concatenate((y_bin, gen_lbl, y_real))
            else:
                X_ft = np.concatenate((X_sq, gen_X_seq))
                y_ft = np.concatenate((y_bin, gen_lbl))
            
            # Train
            cw = class_weight.compute_class_weight('balanced', classes=np.unique(y_ft), y=y_ft)
            cw_dict = dict(enumerate(cw))
            if 0 in cw_dict: cw_dict[0] *= NORMAL_CLASS_BOOST 
            
            model_A.compile(optimizer=opt_A, loss='binary_crossentropy', metrics=['accuracy'])
            for _ in range(ONLINE_EPOCHS): model_A.train_on_batch(X_ft, y_ft, class_weight=cw_dict)
            
            unc_hist.clear()

        add_to_real_buffer(X_sq, y_bin)
        if i % 20 == 0: 
            print(f"Batch {i}/{n_batches} | Acc: {acc:.4f} | F1: {f1:.4f} | Lat: {metrics['lat'][-1]:.2f}ms")

    # FINAL SAVE
    print(f"\n>>> AVG LATENCY: {np.mean(metrics['lat']):.2f} ms <<<")
    
    # 1. Timeline
    plt.figure(figsize=(10,6))
    plt.plot(metrics['acc'], label='Accuracy')
    plt.plot(metrics['f1'], label='F1-Score', alpha=0.7)
    plt.plot(metrics['unc'], label='Uncertainty', alpha=0.3)
    for d in drift_points: plt.axvline(d, color='red', alpha=0.3)
    plt.legend(); plt.savefig(os.path.join(PLOT_PATH, "Generative.png")); plt.close()
    
    # 2. [NEW] Binary Confusion Matrix
    plt.figure(figsize=(8, 6))
    cm_bin = confusion_matrix(y_true_all, y_pred_all)
    sns.heatmap(cm_bin, annot=True, fmt='d', cmap='Blues', xticklabels=['Normal', 'Attack'], yticklabels=['Normal', 'Attack'])
    plt.title("Generative Replay Binary CM"); plt.savefig(os.path.join(PLOT_PATH, "GAN_CM_Binary.png")); plt.close()
    
    try:
        # Save H5 (QUAN TRỌNG)
        model_A.save(os.path.join(MODEL_PATH, "final_model_A_GAN.h5"), save_format='h5')
        model_A.save(os.path.join(MODEL_PATH, "final_model_A_GAN.keras"))
        print("✅ Saved model to .h5 and .keras")
    except Exception as e:
        print(f"❌ Error saving model: {e}")
    
    print("DONE.")

if __name__ == "__main__":
    main()