from data_import import load_all_data
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, rankdata
from pathlib import Path

# =========================
# LOAD DATA
# =========================
X_train, y_train, X_T1, y_T1, X_T2, y_T2 = load_all_data(
    train_mat_path=r"C:\Users\mnaser1\OneDrive - Kennesaw State University\Desktop\ASDSpeech-main - 2\data\training\train_data.mat",
    train_ids_mat_path=r"C:\Users\mnaser1\OneDrive - Kennesaw State University\Desktop\ASDSpeech-main - 2\data\training\ids_fixed.mat",
    train_xlsx_path=r"C:\Users\mnaser1\OneDrive - Kennesaw State University\Desktop\ASDSpeech-main - 2\data\training\data_train.xlsx",
    test_dir=r"C:\Users\mnaser1\OneDrive - Kennesaw State University\Desktop\ASDSpeech-main - 2\data\testing"
)

# =========================
# FLATTEN WINDOWS
# =========================
X_train_flat = X_train.reshape(X_train.shape[0], -1, X_train.shape[-1])  # (N,1000,49)
X_T1_flat    = X_T1.reshape(X_T1.shape[0], -1, X_T1.shape[-1])
X_T2_flat    = X_T2.reshape(X_T2.shape[0], -1, X_T2.shape[-1])

# combined (train + tests)
X_all_flat = np.concatenate([X_train_flat, X_T1_flat, X_T2_flat], axis=0)
y_all      = np.concatenate([y_train, y_T1, y_T2], axis=0)

N, W, F = X_train_flat.shape
n_perm = 1000

out_dir = Path(r"C:\Users\mnaser1\OneDrive - Kennesaw State University\Desktop\ASDSpeech-main - 2")
out_dir.mkdir(parents=True, exist_ok=True)

def compute_stats(x_feat_1d, y_col_1d, seed=0):
    x_feat_1d = np.asarray(x_feat_1d, dtype=float)
    y_col_1d  = np.asarray(y_col_1d,  dtype=float)

    mask = np.isfinite(x_feat_1d) & np.isfinite(y_col_1d)
    x = x_feat_1d[mask]
    y = y_col_1d[mask]

    r_obs, _ = pearsonr(x, y)

    rng = np.random.default_rng(seed)
    r_null = np.empty(n_perm)
    for i in range(n_perm):
        r_null[i], _ = pearsonr(x, rng.permutation(y))

    thr = np.percentile(np.abs(r_null), 97.5)
    p_emp = (np.sum(np.abs(r_null) >= np.abs(r_obs)) + 1) / (n_perm + 1)
    sig = np.abs(r_obs) > thr

    rho, p_s = pearsonr(rankdata(x), rankdata(y))  # "Spearman" via ranks

    return r_obs, p_emp, sig, rho, p_s, len(x)

# =========================
# LOOP OVER y COLUMNS (START FROM 1)
# =========================
for c in range(2, y_train.shape[1]):  # skip first column
    y_train_col = y_train[:, c]
    y_all_col   = y_all[:, c]

    rows = []

    for j in range(F):
        # one value per subject: mean over 1000 windows
        x_train_feat = X_train_flat[:, :, j].mean(axis=1)
        x_all_feat   = X_all_flat[:, :, j].mean(axis=1)

        # train-only stats
        r_tr, pemp_tr, sig_tr, rho_tr, ps_tr, n_tr = compute_stats(
            x_train_feat, y_train_col, seed=1000*c + j
        )

        # combined stats
        r_all, pemp_all, sig_all, rho_all, ps_all, n_all = compute_stats(
            x_all_feat, y_all_col, seed=2000*c + j
        )

        rows.append({
            "feature_idx": j,

            "train_pearson_r": r_tr,
            "train_pearson_p_emp": pemp_tr,
            "train_pearson_sig": sig_tr,
            "train_spearman_rho": rho_tr,
            "train_spearman_p": ps_tr,
            "train_n_used": n_tr,

            "all_pearson_r": r_all,
            "all_pearson_p_emp": pemp_all,
            "all_pearson_sig": sig_all,
            "all_spearman_rho": rho_all,
            "all_spearman_p": ps_all,
            "all_n_used": n_all,
        })

    df_results = pd.DataFrame(rows)
    out_path = out_dir / f"feature_correlation_results_ycol{c}.csv"
    df_results.to_csv(out_path, index=False)

    print(f"Saved: {out_path}")