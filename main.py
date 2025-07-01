# Load
import os
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import numpy as np
from sklearn.metrics import accuracy_score
from padelpy import padeldescriptor
from rdkit.Chem import AllChem as Chem
import custom_preprocessing as cp

def canonical_smiles(df, smiles_column):
    df['canonical_smiles'] = df[smiles_column].apply(lambda x: Chem.MolToSmiles(Chem.MolFromSmiles(x)))
    return df

def compute_fps(df, fingerprint="MACCS", path="fingerprints"):
    # Only use the specified fingerprint
    xml_file = os.path.join(path, f"{fingerprint}.xml")
    if not os.path.exists(xml_file):
        raise FileNotFoundError(f"{xml_file} not found.")
    df['canonical_smiles'].to_csv(os.path.join(path, 'smiles.smi'), sep='\t', index=False, header=False)
    # Calculate fingerprint
    padeldescriptor(
        mol_dir=os.path.join(path, 'smiles.smi'),
        d_file=os.path.join(path, f'{fingerprint}.csv'),
        descriptortypes=xml_file,
        retainorder=True,
        removesalt=True,
        threads=2,
        detectaromaticity=True,
        standardizetautomers=True,
        standardizenitro=True,
        fingerprints=True
    )
    Fingerprint = pd.read_csv(os.path.join(path, f'{fingerprint}.csv')).set_index(df.index)
    if 'Name' in Fingerprint.columns:
        Fingerprint = Fingerprint.drop('Name', axis=1)
    Fingerprint.to_csv(os.path.join(path, f'{fingerprint}.csv'))
    print(f'{fingerprint}.csv done')
    fp_ma = pd.read_csv(os.path.join(path, f'{fingerprint}.csv'), index_col=0)
    return fp_ma

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

def load_model_and_scaler(model_path):
    model, scaler = joblib.load(model_path)
    return model, scaler

def predict_with_model(model, scaler, data):
    data_scaled = scaler.transform(data)
    predictions = model.predict(data_scaled)
    return predictions

def main(fingerprint="MACCS", data_dir="Hepa", excel_file="xanthine.xlsx", name="hepa"):                                                         

    # load the smiles and class of excel_file
    df = pd.read_excel(excel_file, index_col=0)
    df = cp.canonical_smiles(df, "SMILES")
    df = cp.remove_inorganic(df, "canonical_smiles")
    df = cp.remove_mixtures(df, "canonical_smiles")
    model_path = os.path.join("models", f"{fingerprint.lower()}_model_{name}.joblib")
    x_train = pd.read_csv(os.path.join(data_dir, "train", f"{fingerprint.lower()}_{name}.csv"), index_col=0)
    df_clean = df.copy()
    x_test = compute_fps(df_clean, fingerprint=fingerprint, path="fingerprints")
    print("X-test features", x_test.shape)
    # load x_train for fingerprint matching
    x_train = pd.read_csv(os.path.join(data_dir, "train", f"{fingerprint}_{name}.csv"), index_col=0)
    x_train = remove_constant_string_des(x_train)
    x_train = remove_highly_correlated_features(x_train, threshold=0.7)
    print("X-train features", x_train.shape)
    x_test = x_test.loc[:, x_train.columns]
    print("X-test features after matching with x_train", x_test.shape)
    # Load model and scaler
    model, scaler = load_model_and_scaler(model_path)
    # predict
    predictions = predict_with_model(model, scaler, x_test)
    # get csv file with predictions
    df_clean["Predicted_Class"] = predictions
    df_clean.to_csv(os.path.join("predictions", f"predictions_{os.path.splitext(os.path.basename(excel_file))[0]}.csv"))
    return df_clean
if __name__ == "__main__":
    # Example usage: main(fingerprint="MACCS", data_dir="Hepa", excel_file="xanthine.xlsx")
    main(fingerprint="pubchem", data_dir="hepa", excel_file="D_KB.xlsx")
    main(fingerprint="krfpc",   data_dir="neu", excel_file="D_KB.xlsx")
    main(fingerprint="ap2dc",  data_dir="pbmc", excel_file="D_KB.xlsx")
    main(fingerprint="krfp",   data_dir="renal", excel_file="D_KB.xlsx")
    main(fingerprint="subfp", data_dir="res", excel_file="D_KB.xlsx")
    main(fingerprint="subfp", data_dir="scar", excel_file="D_KB.xlsx")
    main(fingerprint="supfpc", data_dir="car", excel_file="D_KB.xlsx")
    main(fingerprint="pubchem", data_dir="skin", excel_file="D_KB.xlsx")

