from data_import import load_all_data
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, rankdata

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
y_col = np.asarray(y_train[:, 3], dtype=float)  # target column

N, W, F = X_train_flat.shape
n_perm = 1000
rng = np.random.default_rng(0)

rows = []

# =========================
# LOOP OVER 49 FEATURES
# =========================
for j in range(F):
    x_feat = X_train_flat[:, :, j].mean(axis=1).astype(float)

    mask = np.isfinite(x_feat) & np.isfinite(y_col)
    x = x_feat[mask]
    y = y_col[mask]

    # Pearson (observed)
    r_obs, _ = pearsonr(x, y)

    # permutation test
    r_null = np.empty(n_perm)
    for i in range(n_perm):
        r_null[i], _ = pearsonr(x, rng.permutation(y))

    thr = np.percentile(np.abs(r_null), 97.5)
    p_emp = (np.sum(np.abs(r_null) >= np.abs(r_obs)) + 1) / (n_perm + 1)
    sig = np.abs(r_obs) > thr

    # Spearman = Pearson on ranks
    rho, p_s = pearsonr(rankdata(x), rankdata(y))

    rows.append({
        "feature_idx": j,
        "pearson_r": r_obs,
        "pearson_p_emp": p_emp,
        "pearson_sig": sig,
        "spearman_rho": rho,
        "spearman_p": p_s,
        "n_used": len(x)
    })

# =========================
# SAVE TO CSV
# =========================
df_results = pd.DataFrame(rows)
out_path = fr"C:\Users\mnaser1\OneDrive - Kennesaw State University\Desktop\ASDSpeech-main - 2\feature_correlation_results.csv"
df_results.to_csv(out_path, index=False)

print(f"Saved results to: {out_path}")
print(df_results.head())
