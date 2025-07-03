import pandas as pd
from rdkit import Chem

# Đọc dữ liệu gốc
df = pd.read_csv("training_data/x_train_Car.csv")

# Hàm kiểm tra SMILES hợp lệ
def is_valid_smiles(smiles):
    try:
        mol = Chem.MolFromSmiles(str(smiles))
        return mol is not None
    except:
        return False

# Áp dụng kiểm tra
df["is_valid"] = df["SMILES"].astype(str).apply(is_valid_smiles)

# SMILES lỗi
invalid_df = df[~df["is_valid"]]
print(f"❌ Số SMILES lỗi: {len(invalid_df)}")
print(invalid_df[["SMILES"]])

# (Tuỳ chọn) Ghi các dòng lỗi ra file CSV để xử lý bằng tay
invalid_df.to_csv("invalid_smiles_car.csv", index=False)
valid_df = df[df["is_valid"]].copy()
valid_df.to_csv("training_data/x_train_Car_cleaned.csv", index=False)

