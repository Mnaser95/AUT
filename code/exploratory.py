from data_import import load_all_data
import numpy as np
from scipy.stats import rankdata, pearsonr
import matplotlib.pyplot as plt


X_train, y_train, X_T1, y_T1, X_T2, y_T2 = load_all_data(
    train_mat_path=r"C:\Users\mnaser1\OneDrive - Kennesaw State University\Desktop\ASDSpeech-main - 2\data\training\train_data.mat",
    train_ids_mat_path=r"C:\Users\mnaser1\OneDrive - Kennesaw State University\Desktop\ASDSpeech-main - 2\data\training\ids_fixed.mat",
    train_xlsx_path=r"C:\Users\mnaser1\OneDrive - Kennesaw State University\Desktop\ASDSpeech-main - 2\data\training\data_train.xlsx",
    test_dir=r"C:\Users\mnaser1\OneDrive - Kennesaw State University\Desktop\ASDSpeech-main - 2\data\testing"
)

X_train_flat = X_train.reshape(X_train.shape[0], -1, X_train.shape[-1])

x_feat = X_train_flat[:, :, 0].mean(axis=1)   # (N,)
y_col  = y_train[:, 3]                        # (N,)

x_feat = np.asarray(x_feat, dtype=float)
y_col  = np.asarray(y_col,  dtype=float)
mask = np.isfinite(x_feat) & np.isfinite(y_col)
x = x_feat[mask]
y = y_col[mask]


# -------------------------
# Pearson (same as before + permutation)
# -------------------------
r_obs, _ = pearsonr(x, y)
print("Observed Pearson r:", r_obs)

rng = np.random.default_rng(0)
n_perm = 1000
r_null = np.empty(n_perm, dtype=float)

for i in range(n_perm):
    y_perm = rng.permutation(y)
    r_null[i], _ = pearsonr(x, y_perm)

thr = np.percentile(np.abs(r_null), 97.5)
sig = np.abs(r_obs) > thr
p_emp = (np.sum(np.abs(r_null) >= np.abs(r_obs)) + 1) / (n_perm + 1)

print("N used:", len(x))
print("Null |r| 97.5% threshold:", thr)
print("Significant?", sig)
print("Empirical p-value:", p_emp)

# -------------------------
# Spearman (no permutation; standard p-value)
# -------------------------
rho, p_s = pearsonr(rankdata(x), rankdata(y))
print("Spearman rho:", rho)
print("Spearman p-value:", p_s)

