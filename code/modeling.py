# cnn_regression_tf_replica_T1eval_T2test.py
# Full replacement: same pipeline you built (NaN removal first, T1 eval, T2 test, Module4 optional)
# but CNN architecture/optimizer/initializers replicate models_class.py (CNN_mdl-like).

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

TARGET_COL_IDX = 2   # third column (0-based)

EXCLUDE_MODULE_4 = True
MODULE_IDX = 10

# Replication flag: match the original pipeline that trains on y/max(y_train)
REPLICATE_Y_SCALING = True

# Training
BATCH_SIZE = 16
LR = 1e-3
EPOCHS = 120
PATIENCE = 12

# Paths
ROOT = Path(r"C:\Users\mnaser1\OneDrive - Kennesaw State University\Desktop\ASDSpeech-main - 2")
TRAIN_MAT = ROOT / r"data\training\train_data.mat"
TRAIN_IDS = ROOT / r"data\training\ids_fixed.mat"
TRAIN_XLSX = ROOT / r"data\training\data_train.xlsx"
TEST_DIR = ROOT / r"data\testing"

OUT_DIR = ROOT / "ml_outputs"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# =========================
# HELPERS
# =========================
def module_is_4(arr):
    """Robust detection of module==4 across mixed dtypes."""
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
    """
    Remove samples (rows) that contain ANY missing values in X or y.
    Works even if y contains strings (ids, gender, etc.).
    """
    X = np.asarray(X)
    y = np.asarray(y)

    X_nan = np.isnan(X).any(axis=tuple(range(1, X.ndim)))  # numeric-only
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
    """
    X: (N, T, F)
    Return mean/std over (N*T, F) computed from TRAIN only.
    """
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


def nrmse(y_true, y_pred, mode="range"):
    """
    NRMSE = RMSE / denom, denom by default is range(y_true).
    """
    y_true = np.asarray(y_true, dtype=np.float32)
    y_pred = np.asarray(y_pred, dtype=np.float32)

    rmse = np.sqrt(np.mean((y_pred - y_true) ** 2))
    if mode == "range":
        denom = np.max(y_true) - np.min(y_true)
    elif mode == "mean":
        denom = np.mean(np.abs(y_true))
    elif mode == "std":
        denom = np.std(y_true)
    else:
        raise ValueError("mode must be 'range', 'mean', or 'std'")
    return float(rmse / (denom + 1e-8))


def evaluate_nrmse(model, X, y, name, y_scale=1.0, out_dir=None, out_prefix=None):
    """
    Predicts, rescales predictions if needed, computes RMSE/NRMSE/MAE,
    and optionally saves prediction CSV.
    """
    y = np.asarray(y, dtype=np.float32).reshape(-1)
    y_pred = model.predict(X, batch_size=BATCH_SIZE, verbose=0).reshape(-1).astype(np.float32)

    # If you trained on normalized y, rescale back here
    y_pred = y_pred * float(y_scale)

    rmse_val = float(np.sqrt(np.mean((y_pred - y) ** 2)))
    nrmse_val = nrmse(y, y_pred, mode="range")
    mae_val = float(np.mean(np.abs(y_pred - y)))

    print(f"{name} → RMSE={rmse_val:.4f}, NRMSE={nrmse_val:.4f}, MAE={mae_val:.4f}")

    if out_dir is not None and out_prefix is not None:
        out_path = Path(out_dir) / f"{out_prefix}.csv"
        pd.DataFrame({"y_true": y, "y_pred": y_pred, "error": y_pred - y}).to_csv(out_path, index=False)
        print(f"Saved predictions: {out_path}")

    return {"rmse": rmse_val, "nrmse": nrmse_val, "mae": mae_val}


# =========================
# REPLICA MODEL (matches models_class.py-style CNN)
# =========================
def correlation_metric(y_true, y_pred):
    """
    Similar spirit to their correlation() metric: returns 1 - Pearson r.
    Keeping it for logging (not used for selection by default).
    """
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)

    mx = tf.reduce_mean(y_true)
    my = tf.reduce_mean(y_pred)
    xm = y_true - mx
    ym = y_pred - my

    r_num = tf.reduce_mean(xm * ym)
    r_den = tf.math.reduce_std(xm) * tf.math.reduce_std(ym)
    r = r_num / (r_den + 1e-8)
    return 1.0 - r


def build_cnn_regressor_replica(input_shape, lr=1e-3):
    """
    Replica of the CNN structure they used:
      Conv1D(256,k=3,relu) -> MaxPool1D(3) -> Conv1D(256,k=3,relu)
      -> Dense 1024 -> Dropout(0.5) -> Dense 512 -> Dropout(0.5)
      -> Dense 256 -> Dense 128 -> Flatten -> Dense(1)

    Uses RandomUniform(seed=1) and RMSprop(lr).
    """
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Conv1D, MaxPool1D, Dense, Dropout, Flatten
    from tensorflow.keras.optimizers import RMSprop
    from tensorflow.keras.initializers import RandomUniform

    init = RandomUniform(seed=1)

    model = Sequential()
    model.add(Conv1D(256, 3, activation="relu", input_shape=input_shape, kernel_initializer=init))
    model.add(MaxPool1D(pool_size=3))
    model.add(Conv1D(256, 3, activation="relu", kernel_initializer=init))

    # Dense on 3D applies time-step wise (same behavior as their code)
    model.add(Dense(1024, activation="relu", kernel_initializer=init))
    model.add(Dropout(0.5))
    model.add(Dense(512, activation="relu", kernel_initializer=init))
    model.add(Dropout(0.5))
    model.add(Dense(256, activation="relu", kernel_initializer=init))
    model.add(Dense(128, activation="relu", kernel_initializer=init))

    model.add(Flatten())
    model.add(Dense(1, activation="linear", kernel_initializer=init))

    model.compile(
        loss="mse",
        optimizer=RMSprop(learning_rate=lr),
        metrics=[
            correlation_metric,
            tf.keras.metrics.MSE,
            tf.keras.metrics.RootMeanSquaredError(name="rmse"),
            tf.keras.metrics.MeanAbsoluteError(name="mae"),
        ],
    )
    return model


# =========================
# MAIN
# =========================
def main():
    # -------- load --------
    X_train, y_train, X_T1, y_T1, X_T2, y_T2 = load_all_data(
        train_mat_path=str(TRAIN_MAT),
        train_ids_mat_path=str(TRAIN_IDS),
        train_xlsx_path=str(TRAIN_XLSX),
        test_dir=str(TEST_DIR),
    )

    # -------- REMOVE ALL NaNs FIRST --------
    X_train, y_train = drop_nan_rows(X_train, y_train)
    X_T1, y_T1       = drop_nan_rows(X_T1, y_T1)
    X_T2, y_T2       = drop_nan_rows(X_T2, y_T2)

    # -------- reshape AFTER NaN removal --------
    # (N, 10, 1000, 49) -> (N, 10*1000, 49) or (N, 1000*?, 49) depending on loader
    X_train = X_train.reshape(X_train.shape[0], -1, X_train.shape[-1]).astype(np.float32)
    X_T1    = X_T1.reshape(X_T1.shape[0], -1, X_T1.shape[-1]).astype(np.float32)
    X_T2    = X_T2.reshape(X_T2.shape[0], -1, X_T2.shape[-1]).astype(np.float32)

    # -------- optional module 4 filtering (TRAIN only) --------
    X_train, y_train = apply_module4_filter(X_train, y_train)

    # -------- target extraction (3rd column) --------
    ytr = to_float_1d(y_train[:, TARGET_COL_IDX])
    yv  = to_float_1d(y_T1[:, TARGET_COL_IDX])
    yt  = to_float_1d(y_T2[:, TARGET_COL_IDX])

    # If conversion created NaNs (strings -> NaN), drop per split
    tr_ok = np.isfinite(ytr)
    v_ok  = np.isfinite(yv)
    t_ok  = np.isfinite(yt)

    X_train, ytr = X_train[tr_ok], ytr[tr_ok]
    X_T1,    yv  = X_T1[v_ok], yv[v_ok]
    X_T2,    yt  = X_T2[t_ok], yt[t_ok]

    # -------- normalize X using TRAIN ONLY (keep your working choice) --------
    mean, std = zscore_fit(X_train)
    X_train = zscore_apply(X_train, mean, std)
    X_T1    = zscore_apply(X_T1, mean, std)
    X_T2    = zscore_apply(X_T2, mean, std)

    # -------- replicate y scaling (optional) --------
    # Train on y / max(y_train), and rescale predictions back by y_scale during evaluation.
    y_scale = 1.0
    if REPLICATE_Y_SCALING:
        y_scale = float(np.max(ytr))
        ytr_fit = (ytr / (y_scale + 1e-8)).astype(np.float32)
        yv_fit  = (yv  / (y_scale + 1e-8)).astype(np.float32)
    else:
        ytr_fit = ytr.astype(np.float32)
        yv_fit  = yv.astype(np.float32)

    # -------- datasets --------
    ds_train = make_ds(X_train, ytr_fit, BATCH_SIZE, shuffle=True)
    ds_val   = make_ds(X_T1,   yv_fit,  BATCH_SIZE, shuffle=False)  # T1 eval/val

    # -------- model (REPLICA) --------
    model = build_cnn_regressor_replica(
        input_shape=(X_train.shape[1], X_train.shape[2]),
        lr=LR
    )

    module_tag = "noModule4" if EXCLUDE_MODULE_4 else "allModules"
    yscale_tag = "yScaled" if REPLICATE_Y_SCALING else "yRaw"
    ckpt_path = OUT_DIR / f"best_tf_CNNreplica_targetCol{TARGET_COL_IDX}_{module_tag}_{yscale_tag}.keras"

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(ckpt_path),
            monitor="val_rmse",
            mode="min",
            save_best_only=True,
            save_weights_only=False,
            verbose=1,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_rmse",
            mode="min",
            patience=PATIENCE,
            restore_best_weights=True,
            verbose=1,
        ),
        tf.keras.callbacks.CSVLogger(
            str(OUT_DIR / f"history_tf_CNNreplica_targetCol{TARGET_COL_IDX}_{module_tag}_{yscale_tag}.csv"),
            append=False,
        ),
    ]

    # -------- train (T1 used for validation/early-stopping) --------
    model.fit(
        ds_train,
        validation_data=ds_val,
        epochs=EPOCHS,
        callbacks=callbacks,
        verbose=1,
    )

    # =========================
    # EVALUATION (NRMSE) ON T1 and T2 (in original y units)
    # =========================
    res_T1 = evaluate_nrmse(
        model, X_T1, yv, "T1 (eval)",
        y_scale=y_scale,
        out_dir=OUT_DIR,
        out_prefix=f"preds_T1_eval_CNNreplica_targetCol{TARGET_COL_IDX}_{module_tag}_{yscale_tag}"
    )

    res_T2 = evaluate_nrmse(
        model, X_T2, yt, "T2 (test)",
        y_scale=y_scale,
        out_dir=OUT_DIR,
        out_prefix=f"preds_T2_test_CNNreplica_targetCol{TARGET_COL_IDX}_{module_tag}_{yscale_tag}"
    )

    # Save metrics CSV
    metrics_csv = OUT_DIR / f"metrics_CNNreplica_targetCol{TARGET_COL_IDX}_{module_tag}_{yscale_tag}.csv"
    pd.DataFrame({
        "split": ["T1", "T2"],
        "rmse": [res_T1["rmse"], res_T2["rmse"]],
        "nrmse_range": [res_T1["nrmse"], res_T2["nrmse"]],
        "mae": [res_T1["mae"], res_T2["mae"]],
        "y_scale_used": [y_scale, y_scale],
        "replicate_y_scaling": [REPLICATE_Y_SCALING, REPLICATE_Y_SCALING],
    }).to_csv(metrics_csv, index=False)
    print(f"Saved metrics: {metrics_csv}")

    # Save summary txt
    summary_path = OUT_DIR / f"summary_CNNreplica_targetCol{TARGET_COL_IDX}_{module_tag}_{yscale_tag}.txt"
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(f"TARGET_COL_IDX: {TARGET_COL_IDX}\n")
        f.write(f"EXCLUDE_MODULE_4: {EXCLUDE_MODULE_4}\n")
        f.write(f"REPLICATE_Y_SCALING: {REPLICATE_Y_SCALING}\n")
        f.write(f"y_scale (max train y): {y_scale}\n\n")
        f.write(f"T1: RMSE={res_T1['rmse']:.6f}, NRMSE(range)={res_T1['nrmse']:.6f}, MAE={res_T1['mae']:.6f}\n")
        f.write(f"T2: RMSE={res_T2['rmse']:.6f}, NRMSE(range)={res_T2['nrmse']:.6f}, MAE={res_T2['mae']:.6f}\n")

    print(f"Saved summary: {summary_path}")
    print(f"Saved best model: {ckpt_path}")


if __name__ == "__main__":
    main()
