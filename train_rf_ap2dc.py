import pandas as pd
import joblib
import subprocess
import uuid
import os

from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

# ===== CUSTOM FUNCTIONS (you must define in custom_preprocessing.py) =====
from custom_preprocessing import (
    canonical_smiles,
    remove_inorganic,
    remove_mixtures,
    remove_constant_string_des,
    remove_highly_correlated_features
)

# ===== FUNCTION TO CALCULATE DESCRIPTORS =====
def calculate_descriptors(smiles_list, xml_path):
    uid = str(uuid.uuid4())
    input_file = f"temp_{uid}.smi"
    output_file = f"temp_{uid}.csv"

    # Write SMILES to temp file
    with open(input_file, "w") as f:
        f.write("Name,SMILES\n")
        for i, smi in enumerate(smiles_list):
            f.write(f"Mol_{i},{smi}\n")

    # Run PaDEL
    subprocess.run([
        "java", "-Xms2G", "-Xmx2G", "-jar", "PaDEL-Descriptor.jar",
        "-removesalt", "-standardizenitro", "-fingerprints",
        "-descriptortypes", xml_path,
        "-dir", ".", "-file", output_file, "-2d"
    ], check=True)

    # Load descriptor
    df = pd.read_csv(output_file)

    # Clean up
    os.remove(input_file)
    os.remove(output_file)

    return df

# ===== LOAD & PREPROCESS TRAINING DATA =====
print("📥 Loading data...")
df = pd.read_csv("training_data/endpoint2_data.csv")  # Must contain 'SMILES' and 'label'

# Canonicalize and clean SMILES
df = canonical_smiles(df, "SMILES")
df = remove_inorganic(df, "canonical_smiles")
df = remove_mixtures(df, "canonical_smiles")

# Extract clean SMILES and labels
smiles = df["canonical_smiles"].tolist()
labels = df["label"].tolist()

# ===== DESCRIPTOR CALCULATION =====
print("🧮 Calculating descriptors...")
desc_df = calculate_descriptors(smiles, "descriptor_xml/AP2D.xml")
X = desc_df.iloc[:, 1:]  # Drop 'Name'
y = labels

# ===== FEATURE SELECTION =====
print("🧹 Cleaning features...")
X = remove_constant_string_des(X)
X = remove_highly_correlated_features(X, threshold=0.7)

# Save feature names
selected_features = X.columns.tolist()
joblib.dump(selected_features, "models/selected_features_ap2dc.joblib")

# ===== MODEL TRAINING =====
print("🧠 Training Random Forest...")
pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="mean")),
    ("rf", RandomForestClassifier(n_estimators=100, random_state=42))
])

pipeline.fit(X, y)

# ===== SAVE MODEL =====
joblib.dump(pipeline, "models/rf_ap2dc.joblib")
print("✅ Model saved to: models/rf_ap2dc.joblib")
print("✅ Selected features saved to: models/selected_features_ap2dc.joblib")
