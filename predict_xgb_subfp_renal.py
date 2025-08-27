import os
import pandas as pd
import joblib
import uuid
from padelpy import padeldescriptor

from custom_preprocessing import (
    canonical_smiles,
    remove_inorganic,
    remove_mixtures
)

# ==== Hàm tính fingerprint (PubChem) ====
def compute_fps(df, fingerprint="SubFP", path="descriptor_xml"):
    xml_file = os.path.join(path, f"{fingerprint}.xml")
    if not os.path.exists(xml_file):
        raise FileNotFoundError(f"{xml_file} not found.")
    
    uid = str(uuid.uuid4())
    smiles_file = f"temp_{uid}.smi"
    output_csv = f"temp_{uid}.csv"

    df["canonical_smiles"].to_csv(smiles_file, sep="\t", index=False, header=False)

    padeldescriptor(
        mol_dir=smiles_file,
        d_file=output_csv,
        descriptortypes=xml_file,
        retainorder=True,
        removesalt=True,
        threads=2,
        detectaromaticity=True,
        standardizetautomers=True,
        standardizenitro=True,
        fingerprints=True
    )

    descriptors_df = pd.read_csv(output_csv)

    # Xóa file tạm
    os.remove(smiles_file)
    os.remove(output_csv)

    return descriptors_df

# ==== Hàm chính để dự đoán từ SMILES ====
def predict_from_smiles(smiles_list):
    # Tạo DataFrame
    df = pd.DataFrame({"SMILES": smiles_list})
    
    # Tiền xử lý
    df = canonical_smiles(df, "SMILES")
    df = remove_inorganic(df, "canonical_smiles")
    df = remove_mixtures(df, "canonical_smiles")

    if df.empty:
        raise ValueError("❌ No valid organic SMILES after preprocessing.")
    
    # Tính descriptor
    desc_df = compute_fps(df, fingerprint="SubFP")
    X = desc_df.select_dtypes(include=["number"])

    # Load model
    model_path = "models/xgb_subfp_renal.joblib"
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"❌ Model not found at: {model_path}")
    
    model = joblib.load(model_path)

    # Dự đoán
    probs = model.predict_proba(X)[:, 1]
    labels = (probs >= 0.5).astype(int)
    results = pd.DataFrame({
        "SMILES": df["canonical_smiles"],
        "Probability": probs,
        "Prediction": ["Toxic" if l == 1 else "Non-toxic" for l in labels]
    })

    return results

# ==== Ví dụ chạy thử nếu chạy script trực tiếp ====
if __name__ == "__main__":
    test_smiles = [
        "CCO",
        "c1ccccc1C(=O)O",  # benzoic acid
        "CN(C)C=O"         # dimethylformamide
    ]
    results = predict_from_smiles(test_smiles)
    print(results)
