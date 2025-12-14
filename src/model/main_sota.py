import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns 
# [UPDATE] Thêm metrics
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, precision_score, recall_score
from sklearn.utils import class_weight 
from collections import deque
import random
import os
import time
import shap 

# Tắt log rác
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# ==========================================
# 1. CẤU HÌNH
# ==========================================
DATA_PATH = "../../dataset/processed/processedstreamvs2.4"
PLOT_PATH = "../../results/plots" 
MODEL_PATH = "../../results/models"

if not os.path.exists(PLOT_PATH): os.makedirs(PLOT_PATH)
if not os.path.exists(MODEL_PATH): os.makedirs(MODEL_PATH)

INITIAL_TRAIN_FILE = os.path.join(DATA_PATH, "processed_initial_train_balanced.parquet")
ONLINE_STREAM_FILE = os.path.join(DATA_PATH, "processed_online_stream.parquet")

# HYPERPARAMETERS
TIME_STEPS = 10 
N_CLASSES_MULTI = 8 
BATCH_SIZE_INIT = 512 
BATCH_SIZE_STREAM = 256 
RAW_BATCH_SIZE = BATCH_SIZE_STREAM * TIME_STEPS 
MC_SAMPLES = 10 
REHEARSAL_SAMPLE_SIZE = 4096 
BUFFER_SIZE = 20000
ONLINE_EPOCHS = 20 

# MANUAL EARLY STOPPING
ES_PATIENCE = 3 
ES_MIN_DELTA = 0.001

# ACTIVE LEARNING
LABELING_BUDGET = 0.05   
PSEUDO_CONFIDENCE = 0.90 

# DRIFT
UNCERTAINTY_WINDOW = 50      
NORMAL_CLASS_BOOST = 3.0      

CLASS_NAMES = ['Normal', 'DoS', 'PortScan', 'Botnet', 'BruteForce', 'WebAttack', 'Infiltration', 'DDoS']
FEATURE_NAMES = [] 

# ==========================================
# 2. KIẾN TRÚC TRANSFORMER (SOTA)
# ==========================================
def transformer_encoder(inputs, head_size, num_heads, ff_dim, dropout=0):
    x = layers.LayerNormalization(epsilon=1e-6)(inputs)
    x = layers.MultiHeadAttention(key_dim=head_size, num_heads=num_heads, dropout=dropout)(x, x)
    x = layers.Add()([x, inputs]) 

    res = x + inputs
    x = layers.LayerNormalization(epsilon=1e-6)(x)
    x = layers.Conv1D(filters=ff_dim, kernel_size=1, activation="relu")(x)
    x = layers.Dropout(dropout)(x)
    x = layers.Conv1D(filters=inputs.shape[-1], kernel_size=1)(x)
    return layers.Add()([x, res])

def create_binary_transformer(input_shape):
    inputs = layers.Input(shape=input_shape)
    x = layers.Dense(64)(inputs) 
    x = transformer_encoder(x, head_size=64, num_heads=4, ff_dim=128, dropout=0.2)
    x = transformer_encoder(x, head_size=64, num_heads=4, ff_dim=128, dropout=0.2)
    x = layers.GlobalAveragePooling1D()(x)
    
    class MCDropout(layers.Dropout):
        def call(self, inputs): return super().call(inputs, training=True)
    x = MCDropout(0.2)(x)
    
    x = layers.Dense(64, activation="relu")(x)
    x = MCDropout(0.2)(x)
    outputs = layers.Dense(1, activation="sigmoid")(x)
    return Model(inputs, outputs)

def create_multiclass_transformer(input_shape):
    inputs = layers.Input(shape=input_shape)
    x = layers.Dense(64)(inputs) 
    for _ in range(2):
        x = transformer_encoder(x, head_size=64, num_heads=4, ff_dim=128, dropout=0.2)
    x = layers.GlobalAveragePooling1D()(x)
    
    class MCDropout(layers.Dropout):
        def call(self, inputs): return super().call(inputs, training=True)
    x = MCDropout(0.2)(x)
    
    x = layers.Dense(64, activation="relu")(x)
    x = MCDropout(0.2)(x)
    outputs = layers.Dense(N_CLASSES_MULTI, activation="softmax")(x)
    return Model(inputs, outputs)

# ==========================================
# 3. CÁC MODULE HỖ TRỢ
# ==========================================
class ActiveStrategy:
    def __init__(self, budget_ratio=0.2):
        self.budget_ratio = budget_ratio
        self.total_queried = 0; self.total_pseudo = 0
    def select_samples(self, X_batch, y_true, y_prob, unc):
        n = len(X_batch); n_bud = int(n * self.budget_ratio)
        idx = np.argsort(unc)
        q_idx = idx[-n_bud:]
        X_q, y_q = X_batch[q_idx], y_true[q_idx]
        self.total_queried += len(q_idx)
        rem_idx = idx[:-n_bud]
        conf = np.abs(y_prob[rem_idx] - 0.5) * 2
        p_idx = rem_idx[conf > PSEUDO_CONFIDENCE]
        X_p, y_p = X_batch[p_idx], (y_prob[p_idx] > 0.5).astype(int).flatten()
        self.total_pseudo += len(p_idx)
        if len(X_p) > 0: return np.concatenate([X_q, X_p]), np.concatenate([y_q, y_p])
        else: return X_q, y_q

class DynamicController:
    def __init__(self, base_lr=1e-5, base_thresh=3.0):
        self.base_lr = base_lr; self.base_thresh = base_thresh; self.ema_uncertainty = 0.01 
    def update(self, current_uncertainty):
        self.ema_uncertainty = 0.9 * self.ema_uncertainty + 0.1 * current_uncertainty
        ratio = current_uncertainty / (self.ema_uncertainty + 1e-9)
        new_lr = min(self.base_lr * max(1.0, ratio * 20.0), 0.001) 
        new_thresh = max(1.5, self.base_thresh / max(1.0, np.log1p(ratio)))
        return new_lr, new_thresh

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
            n_h = int(self.max_size*0.6); n_r = self.max_size - n_h
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
        res = []; acts = [k for k in self.bufs if len(self.bufs[k])>0]
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

@tf.function(reduce_retracing=True)
def get_uncertainty(model, x_batch):
    preds = tf.stack([model(x_batch, training=False) for _ in range(MC_SAMPLES)], axis=0)
    return tf.reduce_mean(tf.math.reduce_variance(preds, axis=0))

def explain_drift_shap(model, background_data, drift_data, batch_idx):
    print(f"\n   🔍 [XAI] Running SHAP Batch {batch_idx}...")
    try:
        bg_2d = np.mean(background_data, axis=1)
        drift_2d = np.mean(drift_data[:20], axis=1) 
        def model_predict_wrapper(x_2d):
            x_3d = np.expand_dims(x_2d, axis=1).repeat(TIME_STEPS, axis=1)
            return model.predict(x_3d, verbose=0)
        explainer = shap.KernelExplainer(model_predict_wrapper, bg_2d[:20])
        with np.errstate(divide='ignore', invalid='ignore'):
            shap_values = explainer.shap_values(drift_2d, nsamples=50, silent=True)
        if isinstance(shap_values, list): sv = shap_values[0]
        else: sv = shap_values
        plt.figure(figsize=(10, 6))
        shap.summary_plot(sv, drift_2d, feature_names=FEATURE_NAMES, show=False)
        plt.title(f"Transformer Drift Explanation - Batch {batch_idx}")
        plt.tight_layout(); plt.savefig(os.path.join(PLOT_PATH, f"XAI_Batch_{batch_idx}.png")); plt.close()
    except: pass

# ==========================================
# 6. MAIN
# ==========================================
def main():
    print(">>> STARTING SOTA TRANSFORMER (FULL METRICS) <<<")
    
    # LOAD
    df_init = pd.read_parquet(INITIAL_TRAIN_FILE)
    feat_cols = [c for c in df_init.columns if c not in ['Label', 'Label_Multi']]
    global FEATURE_NAMES; FEATURE_NAMES = feat_cols
    n_features = len(feat_cols)
    X_init = df_init[feat_cols].values; y_init_bin = df_init['Label'].values; y_init_multi = df_init['Label_Multi'].values
    X_seq, y_bin_seq = prepare_sequences(X_init, y_init_bin, TIME_STEPS)
    _, y_multi_seq = prepare_sequences(X_init, y_init_multi, TIME_STEPS)

    # [OPTIMIZATION 1] Dùng AdamW LR 0.001 + ReduceLROnPlateau
    opt_A_init = tf.keras.optimizers.AdamW(learning_rate=0.001, weight_decay=1e-4)
    opt_B_init = tf.keras.optimizers.AdamW(learning_rate=0.001, weight_decay=1e-4)

    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=1, min_lr=1e-6, verbose=1)
    early = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

    # TRAIN PHASE 1
    print(f"\n--- Phase 1: Binary Transformer Training ---")
    model_A = create_binary_transformer((TIME_STEPS, n_features))
    model_A.compile(optimizer=opt_A_init, loss='binary_crossentropy', metrics=['accuracy'])
    cw_A = class_weight.compute_class_weight('balanced', classes=np.unique(y_bin_seq), y=y_bin_seq)
    model_A.fit(X_seq, y_bin_seq, epochs=25, batch_size=BATCH_SIZE_INIT, validation_split=0.1, callbacks=[early, reduce_lr], class_weight=dict(enumerate(cw_A)), verbose=1)
    buffer_A = SmartRehearsalBuffer(BUFFER_SIZE); buffer_A.add(X_seq, y_bin_seq, 0.0) 

    # TRAIN PHASE 2
    print(f"\n--- Phase 2: Multiclass Transformer Training ---")
    model_B = create_multiclass_transformer((TIME_STEPS, n_features))
    model_B.compile(optimizer=opt_B_init, loss=SparseFocalLoss(), metrics=['accuracy'])
    buffer_B = MulticlassRehearsalBuffer(BUFFER_SIZE, N_CLASSES_MULTI)
    idx_atk = np.where(y_multi_seq > 0)[0]
    if len(idx_atk) > 0:
        X_B, y_B = X_seq[idx_atk], y_multi_seq[idx_atk]
        buffer_B.add(X_B, y_B)
        cw_B = class_weight.compute_class_weight('balanced', classes=np.unique(y_B), y=y_B)
        model_B.fit(X_B, y_B, epochs=25, batch_size=BATCH_SIZE_INIT, validation_split=0.1, callbacks=[early, reduce_lr], class_weight=dict(enumerate(cw_B)), verbose=1)

    # [GIAI ĐOẠN 2] STREAMING - RECOMPILE
    print(f"\n>>> START SOTA STREAMING (FULL METRICS) <<<")
    
    opt_A_stream = tf.keras.optimizers.AdamW(learning_rate=1e-5, weight_decay=1e-4)
    opt_B_stream = tf.keras.optimizers.AdamW(learning_rate=1e-5, weight_decay=1e-4)
    model_A.compile(optimizer=opt_A_stream, loss='binary_crossentropy', metrics=['accuracy'])
    model_B.compile(optimizer=opt_B_stream, loss=SparseFocalLoss(), metrics=['accuracy'])
    
    df_stream = pd.read_parquet(ONLINE_STREAM_FILE)
    X_st_raw = df_stream[feat_cols].values; y_st_bin = df_stream['Label'].values; y_st_multi = df_stream['Label_Multi'].values
    unc_hist = deque(maxlen=UNCERTAINTY_WINDOW)
    
    # [UPDATE] Thêm list Metrics
    metrics = {'acc': [], 'unc': [], 'lat': [], 'f1': [], 'prec': [], 'rec': []}
    drift_points = []
    y_true_all, y_pred_all, y_true_m_all, y_pred_m_all = [], [], [], []
    
    controller = DynamicController(base_lr=1e-5, base_thresh=3.0)
    active_strategy = ActiveStrategy(budget_ratio=LABELING_BUDGET)
    n_batches = len(X_st_raw) // RAW_BATCH_SIZE
    
    for i in range(n_batches):
        start = i * RAW_BATCH_SIZE; end = (i+1) * RAW_BATCH_SIZE
        X_curr = X_st_raw[start:end]
        X_sq, y_bin = prepare_sequences(X_curr, y_st_bin[start:end], TIME_STEPS)
        _, y_mul = prepare_sequences(X_curr, y_st_multi[start:end], TIME_STEPS)
        if len(X_sq) == 0: continue

        t0 = time.time(); p_A = model_A.predict(X_sq, verbose=0); t1 = time.time()
        metrics['lat'].append((t1 - t0) * 1000)

        pred_bin = (p_A > 0.5).astype(int).flatten()
        
        # [UPDATE] Tính Full Metrics
        acc = accuracy_score(y_bin, pred_bin)
        f1 = f1_score(y_bin, pred_bin, zero_division=0)
        prec = precision_score(y_bin, pred_bin, zero_division=0)
        rec = recall_score(y_bin, pred_bin, zero_division=0)
        unc = get_uncertainty(model_A, X_sq).numpy()
        
        metrics['acc'].append(acc); metrics['unc'].append(unc)
        metrics['f1'].append(f1); metrics['prec'].append(prec); metrics['rec'].append(rec)
        
        y_true_all.extend(y_bin); y_pred_all.extend(pred_bin)
        
        # UPDATE LR
        dyn_lr, dyn_thresh = controller.update(unc)
        model_A.optimizer.learning_rate = dyn_lr
        model_B.optimizer.learning_rate = dyn_lr
        
        is_drift = False
        if len(unc_hist) == UNCERTAINTY_WINDOW:
            if unc > np.mean(unc_hist) + dyn_thresh * np.std(unc_hist) and unc > 0.001: is_drift = True
        unc_hist.append(unc)
        
        idx_p = np.where(pred_bin == 1)[0]
        pred_m = np.zeros_like(y_mul)
        if len(idx_p) > 0:
            p_B = model_B.predict(X_sq[idx_p], verbose=0)
            pred_m[idx_p] = np.argmax(p_B, axis=1)
        y_true_m_all.extend(y_mul); y_pred_m_all.extend(pred_m)
        
        if is_drift:
            drift_points.append(i)
            X_act, y_act = active_strategy.select_samples(X_sq, y_bin, p_A.flatten(), unc)
            print(f"[DRIFT] Batch {i} | Unc: {unc:.4f} | Lat: {metrics['lat'][-1]:.2f}ms")

            if len(drift_points) == 1 or len(drift_points) % 10 == 0:
                bg_data = buffer_A.get_all_X(limit=20)
                if len(bg_data) > 0: explain_drift_shap(model_A, bg_data, X_sq, i)

            # ADAPT A
            X_old, y_old = buffer_A.get_sample(REHEARSAL_SAMPLE_SIZE)
            if len(X_old) > 0:
                X_ft = np.concatenate((X_act, X_old)); y_ft = np.concatenate((y_act, y_old))
                cw = class_weight.compute_class_weight('balanced', classes=np.unique(y_ft), y=y_ft)
                cw_dict = dict(enumerate(cw))
                if 0 in cw_dict: cw_dict[0] *= NORMAL_CLASS_BOOST 
                
                best_loss = float('inf'); patience_cnt = 0
                for ep in range(ONLINE_EPOCHS):
                    loss = model_A.train_on_batch(X_ft, y_ft, class_weight=cw_dict)
                    if isinstance(loss, list): loss = loss[0]
                    if loss < best_loss - ES_MIN_DELTA: best_loss = loss; patience_cnt = 0
                    else: patience_cnt += 1
                    if patience_cnt >= ES_PATIENCE: break
            
            # ADAPT B
            idx_t = np.where(y_mul > 0)[0]
            if len(idx_t) > 0:
                X_old_B, y_old_B = buffer_B.get_sample(REHEARSAL_SAMPLE_SIZE)
                X_ft_B = np.concatenate((X_sq[idx_t], X_old_B)); y_ft_B = np.concatenate((y_mul[idx_t], y_old_B))
                for ep in range(ONLINE_EPOCHS):
                    model_B.train_on_batch(X_ft_B, y_ft_B)
            
            unc_hist.clear()

        buffer_A.add(X_sq, y_bin, unc)
        if np.any(y_mul > 0): buffer_B.add(X_sq[y_mul > 0], y_mul[y_mul > 0])
        
        if i % 20 == 0: 
            print(f"Batch {i}/{n_batches} | Acc: {acc:.4f} | F1: {f1:.4f} | Lat: {metrics['lat'][-1]:.2f}ms")

    # FINAL SAVE
    avg_lat = np.mean(metrics['lat'])
    print(f"\n>>> TRANSFORMER AVG LATENCY: {avg_lat:.2f} ms <<<")
    
    # Plot Timeline
    plt.figure(figsize=(10,6))
    plt.plot(metrics['acc'], label='Accuracy')
    plt.plot(metrics['f1'], label='F1-Score', alpha=0.7)
    plt.plot(metrics['unc'], label='Uncertainty', alpha=0.3)
    for d in drift_points: plt.axvline(d, color='red', alpha=0.3)
    plt.legend(); plt.savefig(os.path.join(PLOT_PATH, "Metrics_SOTA_sota.png")); plt.close()
    
    # Plot CM
    plt.figure(figsize=(8, 6))
    cm_bin = confusion_matrix(y_true_all, y_pred_all)
    sns.heatmap(cm_bin, annot=True, fmt='d', cmap='Blues', xticklabels=['Normal', 'Attack'], yticklabels=['Normal', 'Attack'])
    plt.title("SOTA Binary CM"); plt.savefig(os.path.join(PLOT_PATH, "SOTA_CM_Binary_sota.png")); plt.close()

    plt.figure(figsize=(12, 10))
    cm_multi = confusion_matrix(y_true_m_all, y_pred_m_all)
    unique_lbls = sorted(list(set(y_true_m_all) | set(y_pred_m_all)))
    tick_labels = [CLASS_NAMES[i] for i in unique_lbls if i < len(CLASS_NAMES)]
    sns.heatmap(cm_multi, annot=True, fmt='d', cmap='Blues', xticklabels=tick_labels, yticklabels=tick_labels)
    plt.title("SOTA Multiclass CM"); plt.savefig(os.path.join(PLOT_PATH, "SOTA_CM_Multiclass_sota.png")); plt.close()
    
    try:
        model_A.save(os.path.join(MODEL_PATH, "transformer_sota_A.h5"), save_format='h5')
        model_A.save(os.path.join(MODEL_PATH, "transformer_sota_A.keras"))
        model_B.save(os.path.join(MODEL_PATH, "transformer_sota_B.h5"), save_format='h5')
        model_B.save(os.path.join(MODEL_PATH, "transformer_sota_B.keras"))
        print("✅ Saved SOTA models successfully.")
    except Exception as e: print(e)
    
    print("DONE.")

if __name__ == "__main__":
    main()