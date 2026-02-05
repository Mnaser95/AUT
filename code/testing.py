import os, glob
import numpy as np
import pandas as pd
from scipy.io import loadmat

def _clean_ids(s):
    return pd.Series(s).astype(str).str.strip().str.replace(".mat", "", regex=False).tolist()

def _cell10_to_arr(cell):
    return np.stack([np.array(cell[i, 0], dtype=np.float32) for i in range(10)], axis=0)  # (10,100,49)

def load_from_excel_and_mats(excel_path, mats_source, feats_take=49, num_mats=10, is_training=False):
    df = pd.read_excel(excel_path)
    df["rec_id"] = _clean_ids(df["rec_id"])

    if is_training:
        data = loadmat(mats_source, squeeze_me=True, struct_as_record=False)
        feats_all = data["features"]
        feats = np.asarray([feats_all[k][:, :feats_take] for k in range(len(feats_all))], dtype=np.float32)
        X_all = feats.reshape(-1, num_mats, feats.shape[1], feats.shape[2])  # (N,10,100,feats_take)
        # align to excel order
        rec_ids_train = _clean_ids(loadmat(ids_mat_path, squeeze_me=True, struct_as_record=False)["rec_id"])
        rid_to_i = {rid: i for i, rid in enumerate(rec_ids_train)}
        keep = [rid_to_i[rid] for rid in df["rec_id"] if rid in rid_to_i]
        X = X_all[keep]
        df = df[df["rec_id"].isin(rec_ids_train)].reset_index(drop=True)
    else:
        mat_map = {os.path.splitext(os.path.basename(p))[0]: p
                   for p in glob.glob(os.path.join(mats_source, "*.mat"))}
        X = np.stack([
            _cell10_to_arr(loadmat(mat_map[rid], squeeze_me=False, struct_as_record=False)[
                [k for k, v in loadmat(mat_map[rid], squeeze_me=False, struct_as_record=False).items()
                 if not k.startswith("__") and isinstance(v, np.ndarray) and v.dtype == object][0]
            ])
            for rid in df["rec_id"] if rid in mat_map
        ], axis=0)

        df = df[df["rec_id"].isin(mat_map.keys())].reset_index(drop=True)

    y = df.to_numpy()
    print("X shape:", X.shape, "| y shape:", y.shape)
    return X, y, df, df["rec_id"].tolist()


# =========================
# TRAINING
# =========================
train_mat_path = r"C:\Users\mnaser1\OneDrive - Kennesaw State University\Desktop\ASDSpeech-main - 2\data\training\train_data.mat"
ids_mat_path   = r"C:\Users\mnaser1\OneDrive - Kennesaw State University\Desktop\ASDSpeech-main - 2\data\training\ids_fixed.mat"
train_xlsx     = r"C:\Users\mnaser1\OneDrive - Kennesaw State University\Desktop\ASDSpeech-main - 2\data\training\data_train.xlsx"

X_train, y_train, df_train, rec_train = load_from_excel_and_mats(
    excel_path=train_xlsx,
    mats_source=train_mat_path,
    feats_take=49,
    num_mats=10,
    is_training=True
)

# =========================
# TESTING (T1 / T2)
# =========================
test_dir  = r"C:\Users\mnaser1\OneDrive - Kennesaw State University\Desktop\ASDSpeech-main - 2\data\testing"
mats_dir  = os.path.join(test_dir, "data")
test_T1   = os.path.join(test_dir, "data_T1.xlsx")
test_T2   = os.path.join(test_dir, "data_T2.xlsx")

X_T1, y_T1, df_T1, rec_T1 = load_from_excel_and_mats(test_T1, mats_dir, is_training=False)
X_T2, y_T2, df_T2, rec_T2 = load_from_excel_and_mats(test_T2, mats_dir, is_training=False)


