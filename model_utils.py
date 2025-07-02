import joblib
import numpy as np
import pandas as pd
from rdkit import Chem
import subprocess
import uuid
import os

# ==== Hàm tính descriptor từ SMILES và file XML qua PaDEL ====
def calculate_features_from_smiles(smiles: str, xml_path: str) -> np.ndarray:
    uid = str(uuid.uuid4())
    input_file = f"temp_{uid}.smi"
    output_file = f"temp_{uid}.csv"

    with open(input_file, "w") as f:
        f.write(f"mol\t{smiles}\n")

    subprocess.run([
        "java", "-jar", "PaDEL-Descriptor.jar",
        "-descriptortypes", xml_path,
        "-smiles", input_file,
        "-output", output_file,
        "-fingerprints", "false",
        "-removesalt", "true",
        "-standardizenitro", "true"
    ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    df = pd.read_csv(output_file)
    features = df.drop(columns=["Name"]).values[0]

    os.remove(input_file)
    os.remove(output_file)

    return features


# ==== Load tất cả mô hình và file XML tương ứng ====
def load_models(model_dir: str, feature_dir: str):
    models = []
    features = []
    for i in range(8):
        models.append(joblib.load(f"{model_dir}/model_{i+1}.joblib"))
        features.append(f"{feature_dir}/feature_{i+1}.xml")
    return models, features


# ==== Dự đoán với 8 mô hình cho 1 SMILES ====
def predict_all(smiles: str, models, features):
    results = []
    for model, xml in zip(models, features):
        feat = calculate_features_from_smiles(smiles, xml).reshape(1, -1)
        prob = model.predict_proba(feat)[0][1]  # xác suất Toxic
        label = "Toxic" if prob > 0.5 else "Non-toxic"
        results.append((label, round(prob, 3)))
    return results
