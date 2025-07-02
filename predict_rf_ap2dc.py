import pandas as pd
import joblib
import subprocess
import uuid
import os

from custom_preprocessing import (
    canonical_smiles,
    remove_inorganic,
    remove_mixtures
)

# ===== FUNCTION TO CALCULATE DESCRIPTORS =====
def calculate_descriptors(smiles_list, xml_path):
    uid = str(uuid.uuid4())
    input_file = f"temp_{uid}.smi"
    output_file = f"temp_{uid}.csv"

    with open(input_file, "w") as f:
        f.write("Name,SMILES\n")
        for i, smi in enumerate(smiles_list):
            f.write(f"Mol_{i},{smi}\n")

    subprocess.run([
        "java", "-Xms2G", "-Xmx2G", "-jar", "PaDEL-Descriptor.jar",
        "-removesalt", "-standardizenitro", "-fingerprints",
        "-descriptortypes", xml_path,
        "-dir", ".", "-file", output_file, "-2d"
    ], check=True)

    df = pd.read_csv(output_file)
    os.remove(input_file)
    os.remove(output_file)

    return df

# ===== EXAMPLE SMILES INPUT (for test/demo) =====
# This should be replaced by your input source (e.g., from user, CSV, or Streamlit)
input_smiles = []
df = pd.DataFrame({"SMILES": input_smiles})

# ===== PREPROCESS SMILES =====
df = canonical_smiles(df, "SMILES")
df = remove_inorganic(df, "canonical_smiles")
df = remove_mixtures(df, "canonical_smiles")

if df.empty:
    print("❌ No valid organic SMILES after cleaning.")
    exit()

smiles_list = df["canonical_smiles"].tolist()

# ===== LOAD MODEL & SELECTED FEATURES =====
model = joblib.load("models/rf_ap2dc.joblib")
selected_features = joblib.load("models/selected_features_ap2dc.joblib")

# ===== DESCRIPTOR CALCULATION =====
print("🧮 Calculating descriptors...")
desc_df = calculate_descriptors(smiles_list, "descriptor_xml/AP2D.xml")
X = desc_df.iloc[:, 1:]

# ===== ALIGN FEATURES =====
missing_features = set(selected_features) - set(X.columns)
if missing_features:
    print(f"❌ Missing expected features in descriptor: {missing_features}")
    exit()

X = X[selected_features]

# ===== PREDICT =====
print("🔍 Predicting toxicity...")
pred_probs = model.predict_proba(X)[:, 1]
pred_labels = ["Toxic" if p >= 0.5 else "Non-toxic" for p in pred_probs]

# ===== OUTPUT RESULT =====
result_df = pd.DataFrame({
    "SMILES": smiles_list,
    "Prediction": pred_labels,
    "Probability": pred_probs
})

print("\n📊 Prediction Result:")
print(result_df)

# Optional: save to file
result_df.to_csv("predict_output_ap2dc.csv", index=False)
print("\n✅ Results saved to: predict_output_ap2dc.csv")
