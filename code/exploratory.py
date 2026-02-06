from data_import import load_all_data
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, rankdata
from pathlib import Path

# =========================
# CONFIG (SET THIS)
# =========================
RUN_ALL_GENDERS = True      # both genders together (CSV)
RUN_BY_GENDER   = True       # male/female separately (Excel sheets)

# NEW: Exclude ADOS Module 4 subjects (based on y[:, 10])
EXCLUDE_MODULE_4 = True

# which y columns to analyze (your current choice starts at 2)
YCOL_START = 2               # 0=rec_id, 1=Gender, 2=Age, ...
N_PERM = 1000

OUT_DIR = Path(r"C:\Users\mnaser1\OneDrive - Kennesaw State University\Desktop\ASDSpeech-main - 2")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Tag filenames so you never mix filtered vs unfiltered runs
MODULE_TAG = "noModule4" if EXCLUDE_MODULE_4 else "allModules"

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
# INDICES IN y
# =========================
GENDER_IDX = 1
MODULE_IDX = 10

def is_male(arr):
    return (arr == "m") | (arr == "M")

def is_female(arr):
    return (arr == "f") | (arr == "F")

def module_is_4(arr):
    """
    Robustly detect module==4 for mixed dtypes:
    - numbers (4, 4.0)
    - numeric strings ("4", "4.0")
    - bytes
    - anything else -> treated as not-4
    """
    a = np.asarray(arr)
    out = np.zeros(a.shape[0], dtype=bool)
    for i, v in enumerate(a):
        if v is None:
            continue

        # bytes -> str
        if isinstance(v, (bytes, bytearray)):
            try:
                v = v.decode("utf-8", errors="ignore")
            except Exception:
                pass

        # numeric compare
        try:
            out[i] = (float(v) == 4.0)
            continue
        except Exception:
            pass

        # string fallback
        try:
            s = str(v).strip()
            out[i] = (s == "4" or s == "4.0")
        except Exception:
            out[i] = False

    return out

def apply_module4_filter(X, y):
    """Drop rows where Module==4 (based on y[:, MODULE_IDX])."""
    if not EXCLUDE_MODULE_4:
        return X, y
    keep = ~module_is_4(y[:, MODULE_IDX])
    return X[keep], y[keep]

# =========================
# FLATTEN WINDOWS
# =========================
X_train_flat = X_train.reshape(X_train.shape[0], -1, X_train.shape[-1])  # (N,1000,49)
X_T1_flat    = X_T1.reshape(X_T1.shape[0], -1, X_T1.shape[-1])
X_T2_flat    = X_T2.reshape(X_T2.shape[0], -1, X_T2.shape[-1])

X_all_flat = np.concatenate([X_train_flat, X_T1_flat, X_T2_flat], axis=0)
y_all      = np.concatenate([y_train, y_T1, y_T2], axis=0)

# =========================
# OPTIONAL: EXCLUDE MODULE 4 (APPLIED CONSISTENTLY)
# =========================
X_train_flat, y_train = apply_module4_filter(X_train_flat, y_train)
X_all_flat,   y_all   = apply_module4_filter(X_all_flat,   y_all)

# refresh dims after filtering
N, W, F = X_train_flat.shape

def compute_stats(x_feat_1d, y_col_1d, seed=0, n_perm=N_PERM):
    x_feat_1d = np.asarray(x_feat_1d, dtype=float)
    y_col_1d  = np.asarray(y_col_1d,  dtype=float)

    mask = np.isfinite(x_feat_1d) & np.isfinite(y_col_1d)
    x = x_feat_1d[mask]
    y = y_col_1d[mask]

    # pearsonr requires >=2 samples and non-constant vectors
    if x.size < 2 or np.all(x == x[0]) or np.all(y == y[0]):
        return np.nan, np.nan, False, np.nan, np.nan, int(x.size)

    r_obs, _ = pearsonr(x, y)

    rng = np.random.default_rng(seed)
    r_null = np.empty(n_perm, dtype=float)
    for i in range(n_perm):
        yp = rng.permutation(y)
        if np.all(yp == yp[0]):
            r_null[i] = 0.0
        else:
            r_null[i], _ = pearsonr(x, yp)

    thr = np.percentile(np.abs(r_null), 97.5)
    p_emp = (np.sum(np.abs(r_null) >= np.abs(r_obs)) + 1) / (n_perm + 1)
    sig = np.abs(r_obs) > thr

    rho, p_s = pearsonr(rankdata(x), rankdata(y))  # Spearman via ranks

    return r_obs, p_emp, sig, rho, p_s, int(x.size)

def run_for_dataset(Xtr, ytr, Xall, yall, y_col_idx, seed_base=0):
    rows = []
    y_train_col = ytr[:, y_col_idx]
    y_all_col   = yall[:, y_col_idx]

    for j in range(F):
        x_train_feat = Xtr[:, :, j].mean(axis=1)
        x_all_feat   = Xall[:, :, j].mean(axis=1)

        r_tr, pemp_tr, sig_tr, rho_tr, ps_tr, n_tr = compute_stats(
            x_train_feat, y_train_col, seed=seed_base + 1000*y_col_idx + j
        )
        r_all, pemp_all, sig_all, rho_all, ps_all, n_all = compute_stats(
            x_all_feat, y_all_col, seed=seed_base + 2000*y_col_idx + j
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

    return pd.DataFrame(rows)

# =========================
# MAIN LOOP OVER LABEL COLUMNS
# =========================
for c in range(YCOL_START, y_train.shape[1]):

    # ---- case 1: both genders together (CSV) ----
    if RUN_ALL_GENDERS:
        df_all = run_for_dataset(X_train_flat, y_train, X_all_flat, y_all, c, seed_base=0)
        out_csv = OUT_DIR / f"feature_correlation_results_ycol{c}_ALL_{MODULE_TAG}.csv"
        df_all.to_csv(out_csv, index=False)
        print(f"Saved (ALL): {out_csv}")

    # ---- case 2: by gender (Excel with two sheets) ----
    if RUN_BY_GENDER:
        tr_m = is_male(y_train[:, GENDER_IDX])
        tr_f = is_female(y_train[:, GENDER_IDX])
        al_m = is_male(y_all[:,   GENDER_IDX])
        al_f = is_female(y_all[:, GENDER_IDX])

        df_m = run_for_dataset(
            X_train_flat[tr_m], y_train[tr_m],
            X_all_flat[al_m],   y_all[al_m],
            c, seed_base=10_000
        )

        df_f = run_for_dataset(
            X_train_flat[tr_f], y_train[tr_f],
            X_all_flat[al_f],   y_all[al_f],
            c, seed_base=20_000
        )

        out_xlsx = OUT_DIR / f"feature_correlation_results_ycol{c}_BY_GENDER_{MODULE_TAG}.xlsx"
        with pd.ExcelWriter(out_xlsx, engine="openpyxl") as writer:
            df_m.to_excel(writer, sheet_name="male", index=False)
            df_f.to_excel(writer, sheet_name="female", index=False)

        print(f"Saved (BY_GENDER): {out_xlsx}")
