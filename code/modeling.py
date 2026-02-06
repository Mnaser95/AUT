# model_family_benchmark.py
# Same pipeline you validated, but lets you swap model families:
#   - "cnn_replica" (current)
#   - "transformer"
#   - "contrastive_simclr"

from data_import import load_all_data
from pathlib import Path
import numpy as np
import pandas as pd
import tensorflow as tf

# =========================
# CONFIG
# =========================
SEED = 42
tf.random.set_seed(SEED)
np.random.seed(SEED)

TARGET_COL_IDX = 3
EXCLUDE_MODULE_4 = False
MODULE_IDX = 10
REPLICATE_Y_SCALING = True

# Choose one:
MODEL_FAMILY = "contrastive_simclr"         # "cnn_replica" | "transformer" | "contrastive_simclr"

# Training
BATCH_SIZE = 16
LR = 1e-3
EPOCHS = 120
PATIENCE = 12

# Contrastive pretrain settings
PRETRAIN_EPOCHS = 50
PRETRAIN_LR = 1e-3
PROJ_DIM = 128
TEMPERATURE = 0.2

ROOT = Path(r"C:\Users\mnaser1\OneDrive - Kennesaw State University\Desktop\ASDSpeech-main - 2")
TRAIN_MAT = ROOT / r"data\training\train_data.mat"
TRAIN_IDS = ROOT / r"data\training\ids_fixed.mat"
TRAIN_XLSX = ROOT / r"data\training\data_train.xlsx"
TEST_DIR = ROOT / r"data\testing"
OUT_DIR = ROOT / "ml_outputs"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# =========================
# HELPERS (same as yours)
# =========================
def module_is_4(arr):
    a = np.asarray(arr)
    out = np.zeros(a.shape[0], dtype=bool)
    for i, v in enumerate(a):
        if v is None:
            continue
        if isinstance(v, (bytes, bytearray)):
            try:
                v = v.decode("utf-8", errors="ignore")
            except Exception:
                pass
        try:
            out[i] = (float(v) == 4.0)
        except Exception:
            out[i] = False
    return out

def drop_nan_rows(X, y):
    X = np.asarray(X)
    y = np.asarray(y)
    X_nan = np.isnan(X).any(axis=tuple(range(1, X.ndim)))
    if y.ndim == 1:
        y_nan = pd.isna(y)
    else:
        y_nan = pd.isna(y).any(axis=1)
    keep = ~(X_nan | y_nan)
    return X[keep], y[keep]

def apply_module4_filter(X, y):
    if not EXCLUDE_MODULE_4:
        return X, y
    keep = ~module_is_4(y[:, MODULE_IDX])
    return X[keep], y[keep]

def to_float_1d(arr):
    return pd.to_numeric(pd.Series(arr), errors="coerce").to_numpy(dtype=np.float32)

def zscore_fit(X):
    xf = X.reshape(-1, X.shape[-1]).astype(np.float32)
    mean = xf.mean(axis=0)
    std = xf.std(axis=0) + 1e-8
    return mean.astype(np.float32), std.astype(np.float32)

def zscore_apply(X, mean, std):
    return ((X - mean) / std).astype(np.float32)

def make_ds(X, y, batch_size, shuffle=False):
    ds = tf.data.Dataset.from_tensor_slices((X, y))
    if shuffle:
        ds = ds.shuffle(min(len(y), 5000), seed=SEED, reshuffle_each_iteration=True)
    return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)

def nrmse(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=np.float32)
    y_pred = np.asarray(y_pred, dtype=np.float32)
    rmse = np.sqrt(np.mean((y_pred - y_true) ** 2))
    denom = (np.max(y_true) - np.min(y_true)) + 1e-8
    return float(rmse / denom)

def evaluate_nrmse(model, X, y, name, y_scale=1.0, out_path=None):
    y = np.asarray(y, dtype=np.float32).reshape(-1)
    y_pred = model.predict(X, batch_size=BATCH_SIZE, verbose=0).reshape(-1).astype(np.float32)
    y_pred = y_pred * float(y_scale)

    rmse = float(np.sqrt(np.mean((y_pred - y) ** 2)))
    mae = float(np.mean(np.abs(y_pred - y)))
    nn = float(nrmse(y, y_pred))

    print(f"{name} → RMSE={rmse:.4f}, NRMSE={nn:.4f}, MAE={mae:.4f}")

    if out_path is not None:
        pd.DataFrame({"y_true": y, "y_pred": y_pred, "error": y_pred - y}).to_csv(out_path, index=False)

    return {"rmse": rmse, "nrmse": nn, "mae": mae}


# =========================
# FAMILY A: CNN replica (your current)
# =========================
def build_cnn_replica(input_shape, lr=1e-3):
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Conv1D, MaxPool1D, Dense, Dropout, Flatten
    from tensorflow.keras.optimizers import RMSprop
    from tensorflow.keras.initializers import RandomUniform

    init = RandomUniform(seed=1)

    model = Sequential([
        Conv1D(256, 3, activation="relu", input_shape=input_shape, kernel_initializer=init),
        MaxPool1D(pool_size=3),
        Conv1D(256, 3, activation="relu", kernel_initializer=init),
        Dense(1024, activation="relu", kernel_initializer=init),
        Dropout(0.5),
        Dense(512, activation="relu", kernel_initializer=init),
        Dropout(0.5),
        Dense(256, activation="relu", kernel_initializer=init),
        Dense(128, activation="relu", kernel_initializer=init),
        Flatten(),
        Dense(1, activation="linear", kernel_initializer=init),
    ])

    model.compile(
        optimizer=RMSprop(learning_rate=lr),
        loss="mse",
        metrics=[tf.keras.metrics.RootMeanSquaredError(name="rmse"),
                 tf.keras.metrics.MeanAbsoluteError(name="mae")]
    )
    return model


# =========================
# FAMILY B: Transformer regressor
# =========================
@tf.keras.utils.register_keras_serializable(package="Custom")
class TransformerBlock(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1, **kwargs):
        super().__init__(**kwargs)
        self.d_model = int(d_model)
        self.num_heads = int(num_heads)
        self.d_ff = int(d_ff)
        self.dropout = float(dropout)

        self.mha = tf.keras.layers.MultiHeadAttention(num_heads=self.num_heads, key_dim=self.d_model)
        self.ffn = tf.keras.Sequential([
            tf.keras.layers.Dense(self.d_ff, activation="relu"),
            tf.keras.layers.Dense(self.d_model),
        ])
        self.ln1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.ln2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.do1 = tf.keras.layers.Dropout(self.dropout)
        self.do2 = tf.keras.layers.Dropout(self.dropout)

    def call(self, x, training=False):
        attn = self.mha(x, x, training=training)
        x = self.ln1(x + self.do1(attn, training=training))
        ffn = self.ffn(x, training=training)
        x = self.ln2(x + self.do2(ffn, training=training))
        return x

    def get_config(self):
        config = super().get_config()
        config.update({
            "d_model": self.d_model,
            "num_heads": self.num_heads,
            "d_ff": self.d_ff,
            "dropout": self.dropout,
        })
        return config


def build_transformer_regressor(input_shape, lr=1e-3, d_model=128, num_heads=4, d_ff=256, n_blocks=3):
    inp = tf.keras.Input(shape=input_shape)  # (T, F)

    # Project features to d_model
    x = tf.keras.layers.Dense(d_model)(inp)

    for _ in range(n_blocks):
        x = TransformerBlock(d_model=d_model, num_heads=num_heads, d_ff=d_ff, dropout=0.1)(x)

    x = tf.keras.layers.GlobalAveragePooling1D()(x)
    x = tf.keras.layers.Dense(256, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    out = tf.keras.layers.Dense(1)(x)

    model = tf.keras.Model(inp, out)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        loss="mse",
        metrics=[tf.keras.metrics.RootMeanSquaredError(name="rmse"),
                 tf.keras.metrics.MeanAbsoluteError(name="mae")]
    )
    return model


# =========================
# FAMILY C: Contrastive (SimCLR-ish) pretraining + regression
# =========================
def augment_time_series(x):
    """
    x: (T, F) float32
    Simple augmentations: jitter + scaling + time masking.
    """
    # jitter
    noise = tf.random.normal(tf.shape(x), mean=0.0, stddev=0.03)
    x1 = x + noise

    # scaling
    scale = tf.random.uniform([], 0.9, 1.1)
    x1 = x1 * scale

    # time mask
    T = tf.shape(x1)[0]
    mask_len = tf.cast(tf.maximum(1, T // 20), tf.int32)
    start = tf.random.uniform([], 0, tf.maximum(1, T - mask_len), dtype=tf.int32)
    mask = tf.concat([
        tf.ones([start, tf.shape(x1)[1]]),
        tf.zeros([mask_len, tf.shape(x1)[1]]),
        tf.ones([T - start - mask_len, tf.shape(x1)[1]])
    ], axis=0)
    x1 = x1 * mask

    return x1

def build_encoder(input_shape, d_model=128):
    inp = tf.keras.Input(shape=input_shape)
    x = tf.keras.layers.Conv1D(128, 7, strides=2, padding="same", activation="relu")(inp)
    x = tf.keras.layers.Conv1D(256, 5, strides=2, padding="same", activation="relu")(x)
    x = tf.keras.layers.Conv1D(256, 3, strides=2, padding="same", activation="relu")(x)
    x = tf.keras.layers.GlobalAveragePooling1D()(x)
    x = tf.keras.layers.Dense(d_model, activation="relu")(x)
    return tf.keras.Model(inp, x, name="encoder")

def build_projection_head(d_in, proj_dim=128):
    inp = tf.keras.Input(shape=(d_in,))
    x = tf.keras.layers.Dense(256, activation="relu")(inp)
    out = tf.keras.layers.Dense(proj_dim)(x)
    return tf.keras.Model(inp, out, name="proj_head")

def nt_xent_loss(z1, z2, temperature=0.2):
    """
    z1, z2: (B, D) normalized
    """
    z1 = tf.math.l2_normalize(z1, axis=1)
    z2 = tf.math.l2_normalize(z2, axis=1)

    logits_12 = tf.matmul(z1, z2, transpose_b=True) / temperature
    logits_21 = tf.matmul(z2, z1, transpose_b=True) / temperature

    B = tf.shape(z1)[0]
    labels = tf.range(B)
    loss_12 = tf.keras.losses.sparse_categorical_crossentropy(labels, logits_12, from_logits=True)
    loss_21 = tf.keras.losses.sparse_categorical_crossentropy(labels, logits_21, from_logits=True)
    return tf.reduce_mean(loss_12 + loss_21) * 0.5

def pretrain_contrastive(X_train, input_shape):
    # Dataset that emits two augmented views
    def map_views(x):
        v1 = augment_time_series(x)
        v2 = augment_time_series(x)
        return v1, v2

    ds = tf.data.Dataset.from_tensor_slices(X_train.astype(np.float32))
    ds = ds.shuffle(min(len(X_train), 5000), seed=SEED, reshuffle_each_iteration=True)
    ds = ds.map(map_views, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

    encoder = build_encoder(input_shape=input_shape, d_model=256)
    proj = build_projection_head(d_in=encoder.output_shape[-1], proj_dim=PROJ_DIM)

    opt = tf.keras.optimizers.Adam(learning_rate=PRETRAIN_LR)

    @tf.function
    def train_step(v1, v2):
        with tf.GradientTape() as tape:
            h1 = encoder(v1, training=True)
            h2 = encoder(v2, training=True)
            z1 = proj(h1, training=True)
            z2 = proj(h2, training=True)
            loss = nt_xent_loss(z1, z2, temperature=TEMPERATURE)
        vars_ = encoder.trainable_variables + proj.trainable_variables
        grads = tape.gradient(loss, vars_)
        opt.apply_gradients(zip(grads, vars_))
        return loss

    for ep in range(1, PRETRAIN_EPOCHS + 1):
        losses = []
        for v1, v2 in ds:
            losses.append(train_step(v1, v2))
        print(f"[Pretrain] Epoch {ep:03d} | loss={tf.reduce_mean(losses).numpy():.4f}")

    return encoder  # return pretrained encoder only

def build_regressor_from_encoder(encoder, input_shape, lr=1e-3, finetune=True):
    inp = tf.keras.Input(shape=input_shape)
    h = encoder(inp, training=finetune)
    x = tf.keras.layers.Dense(256, activation="relu")(h)
    x = tf.keras.layers.Dropout(0.2)(x)
    out = tf.keras.layers.Dense(1)(x)
    model = tf.keras.Model(inp, out)

    encoder.trainable = bool(finetune)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        loss="mse",
        metrics=[tf.keras.metrics.RootMeanSquaredError(name="rmse"),
                 tf.keras.metrics.MeanAbsoluteError(name="mae")]
    )
    return model


# =========================
# MAIN
# =========================
def main():
    X_train, y_train, X_T1, y_T1, X_T2, y_T2 = load_all_data(
        train_mat_path=str(TRAIN_MAT),
        train_ids_mat_path=str(TRAIN_IDS),
        train_xlsx_path=str(TRAIN_XLSX),
        test_dir=str(TEST_DIR),
    )

    # remove NaNs FIRST
    X_train, y_train = drop_nan_rows(X_train, y_train)
    X_T1, y_T1 = drop_nan_rows(X_T1, y_T1)
    X_T2, y_T2 = drop_nan_rows(X_T2, y_T2)

    # reshape
    X_train = X_train.reshape(X_train.shape[0], -1, X_train.shape[-1]).astype(np.float32)
    X_T1 = X_T1.reshape(X_T1.shape[0], -1, X_T1.shape[-1]).astype(np.float32)
    X_T2 = X_T2.reshape(X_T2.shape[0], -1, X_T2.shape[-1]).astype(np.float32)

    # module filter (train only)
    X_train, y_train = apply_module4_filter(X_train, y_train)

    # target
    ytr = to_float_1d(y_train[:, TARGET_COL_IDX])
    yv = to_float_1d(y_T1[:, TARGET_COL_IDX])
    yt = to_float_1d(y_T2[:, TARGET_COL_IDX])

    # drop NaNs from target conversion
    tr_ok = np.isfinite(ytr); v_ok = np.isfinite(yv); t_ok = np.isfinite(yt)
    X_train, ytr = X_train[tr_ok], ytr[tr_ok]
    X_T1, yv = X_T1[v_ok], yv[v_ok]
    X_T2, yt = X_T2[t_ok], yt[t_ok]

    # normalize X using train only
    mean, std = zscore_fit(X_train)
    X_train = zscore_apply(X_train, mean, std)
    X_T1 = zscore_apply(X_T1, mean, std)
    X_T2 = zscore_apply(X_T2, mean, std)

    # y scaling (optional replication)
    y_scale = 1.0
    if REPLICATE_Y_SCALING:
        y_scale = float(np.max(ytr))
        ytr_fit = (ytr / (y_scale + 1e-8)).astype(np.float32)
        yv_fit = (yv / (y_scale + 1e-8)).astype(np.float32)
    else:
        ytr_fit = ytr.astype(np.float32)
        yv_fit = yv.astype(np.float32)

    ds_train = make_ds(X_train, ytr_fit, BATCH_SIZE, shuffle=True)
    ds_val = make_ds(X_T1, yv_fit, BATCH_SIZE, shuffle=False)

    input_shape = (X_train.shape[1], X_train.shape[2])

    # pick family
    if MODEL_FAMILY == "cnn_replica":
        model = build_cnn_replica(input_shape, lr=LR)
    elif MODEL_FAMILY == "transformer":
        model = build_transformer_regressor(input_shape, lr=LR)
    elif MODEL_FAMILY == "contrastive_simclr":
        encoder = pretrain_contrastive(X_train, input_shape=input_shape)
        model = build_regressor_from_encoder(encoder, input_shape=input_shape, lr=LR, finetune=True)
    else:
        raise ValueError(f"Unknown MODEL_FAMILY: {MODEL_FAMILY}")

    tag = f"{MODEL_FAMILY}_col{TARGET_COL_IDX}" + ("_yScaled" if REPLICATE_Y_SCALING else "_yRaw")
    ckpt_path = OUT_DIR / f"best_{tag}.keras"

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(str(ckpt_path), monitor="val_rmse", mode="min",
                                           save_best_only=True, verbose=1),
        tf.keras.callbacks.EarlyStopping(monitor="val_rmse", mode="min",
                                         patience=PATIENCE, restore_best_weights=True, verbose=1),
        tf.keras.callbacks.CSVLogger(str(OUT_DIR / f"history_{tag}.csv"), append=False),
    ]

    model.fit(ds_train, validation_data=ds_val, epochs=EPOCHS, callbacks=callbacks, verbose=1)

    # Evaluate NRMSE on T1 and T2 (always in original y units)
    res_T1 = evaluate_nrmse(
        model, X_T1, yv, "T1 (eval)", y_scale=y_scale,
        out_path=OUT_DIR / f"preds_T1_{tag}.csv"
    )
    res_T2 = evaluate_nrmse(
        model, X_T2, yt, "T2 (test)", y_scale=y_scale,
        out_path=OUT_DIR / f"preds_T2_{tag}.csv"
    )

    # Save metrics
    metrics_path = OUT_DIR / f"metrics_{tag}.csv"
    pd.DataFrame({
        "split": ["T1", "T2"],
        "rmse": [res_T1["rmse"], res_T2["rmse"]],
        "nrmse_range": [res_T1["nrmse"], res_T2["nrmse"]],
        "mae": [res_T1["mae"], res_T2["mae"]],
    }).to_csv(metrics_path, index=False)
    print(f"Saved: {metrics_path}")
    print(f"Saved model: {ckpt_path}")


if __name__ == "__main__":
    main()
