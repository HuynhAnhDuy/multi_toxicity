import pandas as pd
import joblib
import os
import uuid
import xgboost as xgb
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from padelpy import padeldescriptor

from custom_preprocessing import (
    canonical_smiles,
    remove_inorganic,
    remove_mixtures,
    remove_constant_string_des,
    remove_highly_correlated_features
)

# ===== FUNCTION TO CALCULATE DESCRIPTORS USING padelpy =====
def compute_fps(df, fingerprint="AP2DC", path="descriptor_xml"):
    # Only use the specified fingerprint
    xml_file = os.path.join(path, f"{fingerprint}.xml")
    if not os.path.exists(xml_file):
        raise FileNotFoundError(f"{xml_file} not found.")
    
    smiles_file = os.path.join(path, 'smiles.smi')
    output_csv = os.path.join(path, f'{fingerprint}.csv')

    df['canonical_smiles'].to_csv(smiles_file, sep='\t', index=False, header=False)
    
    # Calculate fingerprint
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

    # Load descriptors from output file
    descriptors_df = pd.read_csv(output_csv)
    
    # Xóa file tạm
    os.remove(smiles_file)
    os.remove(output_csv)

    # Gộp lại với nhãn nếu cần, hoặc chỉ return descriptors
    return descriptors_df

# ===== LOAD & CLEAN TRAINING DATA =====
print("📥 Loading data...")
df = pd.read_csv("training_data/x_train_PBMC.csv")  # Must contain 'SMILES' and 'Label' columns

df = canonical_smiles(df, "SMILES")
df = remove_inorganic(df, "canonical_smiles")
df = remove_mixtures(df, "canonical_smiles")
print(f"✅ Cleaned data: {df.shape[0]} samples")

# ===== DESCRIPTOR CALCULATION =====
print("🧮 Calculating descriptors with PaDEL...")
desc_df = compute_fps(df, "AP2DC")
X = desc_df.select_dtypes(include=["number"])
y = df["Label"].tolist()
print(f"✅ Descriptors calculated for {X.shape[0]} samples, {X.shape[1]} features")

# ===== CHECK FEATURES =====
if X.shape[1] == 0:
    raise ValueError("❌ No numeric features found. Check XML file or descriptor output.")

# ===== MODEL TRAINING =====
print("🧠 Training XGBoost with StandardScaler...")
pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="mean")),
    ("scaler", StandardScaler()),
    ("xgb", xgb.XGBClassifier(max_depth=3))
])
pipeline.fit(X, y)

# ===== SAVE MODEL =====
os.makedirs("models", exist_ok=True)
joblib.dump(pipeline, "models/xgb_ap2dc_pbmc.joblib")
print("✅ Model saved to: models/xgb_ap2dc_pbmc.joblib")
