import os
import numpy as np
import pandas as pd
import joblib
import subprocess
import uuid
import custom_preprocessing as cp

# Endpoint tên đầy đủ cho hiển thị
ENDPOINT_NAMES = [
    "Peripheral blood mononuclear cells toxicity",
    "Nephrotoxicity",
    "Neurotoxicity",
    "Hepatotoxicity",
    "Skin Sensitization",
    "Respiratory Toxicity",
    "Severe Cutaneous Adverse Reaction",
    "Cardiotoxicity"
]

# ==== Hàm tính descriptor từ SMILES theo file XML qua PaDEL ====
def calculate_descriptors(smiles: str, xml_path: str) -> pd.DataFrame:
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

    df = pd.read_csv(output_file).drop(columns=["Name"])
    os.remove(input_file)
    os.remove(output_file)

    return df

def remove_constant_string_des(df):
    df = df.select_dtypes(exclude=['object'])
    for column in df.columns:
        if df[column].nunique() == 1:
            df = df.drop(column, axis=1)
    return df

def remove_highly_correlated_features(df, threshold=0.7):
    corr_matrix = df.corr().abs()
    upper = corr_matrix.where(
        pd.DataFrame(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool), 
                     index=corr_matrix.index, columns=corr_matrix.columns)
    )
    to_drop = [column for column in upper.columns if any(upper[column] > threshold)]
    df_dropped = df.drop(columns=to_drop)
    return df_dropped

# ==== Hàm chính dự đoán cho 8 endpoint ====
def predict_all(smiles: str,
                model_dir="models",
                xml_dir="features",
                train_data_dir="train"):
    
    # Chuẩn hoá SMILES đầu vào
    df = pd.DataFrame({"SMILES": [smiles]})
    df = cp.canonical_smiles(df, "SMILES")
    df = cp.remove_inorganic(df, "canonical_smiles")
    df = cp.remove_mixtures(df, "canonical_smiles")

    if df.empty:
        return [(ENDPOINT_NAMES[i], "Invalid SMILES", 0.0) for i in range(8)]

    canonical = df["canonical_smiles"].values[0]
    results = []

    for i in range(8):  # chỉ số từ 0 → 7
        try:
            model_path = os.path.join(model_dir, f"model_{i+1}.joblib")
            xml_path = os.path.join(xml_dir, f"feature_{i+1}.xml")
            x_train_path = os.path.join(train_data_dir, f"train_{i+1}.csv")

            model = joblib.load(model_path)
            x_test = calculate_descriptors(canonical, xml_path)

            x_train = pd.read_csv(x_train_path, index_col=0)
            x_train = remove_constant_string_des(x_train)
            x_train = remove_highly_correlated_features(x_train, threshold=0.7)

            x_test = x_test[x_train.columns]

            prob = model.predict_proba(x_test)[0][1]
            label = "Possible toxicity" if prob > 0.5 else "Non-toxicity"
            results.append((ENDPOINT_NAMES[i], label, round(prob, 3)))
        except Exception as e:
            results.append((ENDPOINT_NAMES[i], f"Error: {e}", 0.0))

    return results
