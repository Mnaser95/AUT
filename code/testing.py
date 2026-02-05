import numpy as np
import pandas as pd
from scipy.io import loadmat
from pathlib import Path

def load_training_data(train_mat_path, ids_mat_path, meta_xlsx_path, feats_take=49):
    data = loadmat(train_mat_path, squeeze_me=True, struct_as_record=False)
    ids  = loadmat(ids_mat_path,  squeeze_me=True, struct_as_record=False)

    rec_ids = [str(x).replace(".mat", "") for x in ids["rec_id"]]

    df = pd.read_excel(meta_xlsx_path)
    df["rec_id"] = df["rec_id"].astype(str).str.replace(".mat", "")
    df = df[df["rec_id"].isin(rec_ids)]
    df["rec_id"] = pd.Categorical(df["rec_id"], rec_ids, ordered=True)
    df = df.sort_values("rec_id").reset_index(drop=True)

    feats_all = data["features"]
    num_mats = 10

    feats = np.asarray([feats_all[k][:, :feats_take] for k in range(len(feats_all))], dtype=np.float32)
    X = feats.reshape(-1, num_mats, feats.shape[1], feats.shape[2])

    X = X[[rec_ids.index(rid) for rid in df["rec_id"]]]

    y = df.to_numpy()

    print("X shape:", X.shape)
    print("y shape:", y.shape)
    
    return X, y, df, df["rec_id"].tolist()

if __name__ == "__main__":
    X, y, df, rec_ids = load_training_data(
        train_mat_path=r"C:\Users\mnaser1\OneDrive - Kennesaw State University\Desktop\ASDSpeech-main - 2\data\training\train_data.mat",
        ids_mat_path=r"C:\Users\mnaser1\OneDrive - Kennesaw State University\Desktop\ASDSpeech-main - 2\data\training\ids_fixed.mat",
        meta_xlsx_path=r"C:\Users\mnaser1\OneDrive - Kennesaw State University\Desktop\ASDSpeech-main - 2\data\training\data_train.xlsx"
    )




# import os, glob
# import numpy as np
# import pandas as pd
# from scipy.io import loadmat

# # paths
# test_dir  = r"C:\Users\mnaser1\OneDrive - Kennesaw State University\Desktop\ASDSpeech-main - 2\data\testing"
# mats_dir  = os.path.join(test_dir, "data")          # folder that contains feature .mat files
# excel_T1  = os.path.join(test_dir, "data_T1.xlsx")
# excel_T2  = os.path.join(test_dir, "data_T2.xlsx")

# # map available .mat files once
# mat_map = {os.path.splitext(os.path.basename(p))[0]: p
#            for p in glob.glob(os.path.join(mats_dir, "*.mat"))}

# def load_split(excel_path):
#     meta = pd.read_excel(excel_path)
#     meta["rec_id"] = meta["rec_id"].astype(str).str.strip().str.replace(".mat", "", regex=False)

#     X, rows = [], []
#     for _, row in meta.iterrows():
#         rid = row["rec_id"]
#         if rid not in mat_map:
#             continue

#         md = loadmat(mat_map[rid], squeeze_me=False, struct_as_record=False)
#         var = [k for k,v in md.items()
#                if not k.startswith("__") and isinstance(v, np.ndarray) and v.dtype == object][0]
#         cell = md[var]
#         x = np.stack([np.array(cell[i,0], dtype=np.float32) for i in range(10)], axis=0)  # (10,100,49)

#         X.append(x)
#         rows.append(row)

#     return np.stack(X, axis=0), pd.DataFrame(rows).reset_index(drop=True)

# # load both
# X_T1, meta_T1 = load_split(excel_T1)
# X_T2, meta_T2 = load_split(excel_T2)

# print("X_T1:", X_T1.shape, "meta_T1:", meta_T1.shape)
# print("X_T2:", X_T2.shape, "meta_T2:", meta_T2.shape)
