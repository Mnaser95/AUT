from data_import import load_all_data

X_train, y_train, X_T1, y_T1, X_T2, y_T2 = load_all_data(
    train_mat_path=r"C:\Users\mnaser1\OneDrive - Kennesaw State University\Desktop\ASDSpeech-main - 2\data\training\train_data.mat",
    train_ids_mat_path=r"C:\Users\mnaser1\OneDrive - Kennesaw State University\Desktop\ASDSpeech-main - 2\data\training\ids_fixed.mat",
    train_xlsx_path=r"C:\Users\mnaser1\OneDrive - Kennesaw State University\Desktop\ASDSpeech-main - 2\data\training\data_train.xlsx",
    test_dir=r"C:\Users\mnaser1\OneDrive - Kennesaw State University\Desktop\ASDSpeech-main - 2\data\testing"
)

print("TRAIN:", X_train.shape, y_train.shape)
print("TEST T1:", X_T1.shape, y_T1.shape)
print("TEST T2:", X_T2.shape, y_T2.shape)
