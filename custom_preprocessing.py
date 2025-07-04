import pandas as pd
import numpy as np
from rdkit.Chem import AllChem as Chem

# ======= Chuẩn hóa SMILES =======
def canonical_smiles(df, smiles_col):
    df['canonical_smiles'] = df[smiles_col].apply(
        lambda x: Chem.MolToSmiles(Chem.MolFromSmiles(x), isomericSmiles=True)
    )
    return df

# ======= Loại hợp chất vô cơ =======
def has_carbon_atoms(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol:
        return any(atom.GetAtomicNum() == 6 for atom in mol.GetAtoms())
    return False

def remove_inorganic(df, smiles_col):
    df['has_carbon'] = df[smiles_col].apply(has_carbon_atoms)
    df = df[df['has_carbon']]
    return df.drop(columns=['has_carbon'])

# ======= Loại hỗn hợp (chứa dấu chấm) =======
def remove_mixtures(df, smiles_col):
    df = df[~df[smiles_col].str.contains(r'\.')]
    return df

# ======= Xoá đặc trưng kiểu chuỗi và đặc trưng hằng số =======
def remove_constant_string_des(df):
    df = df.select_dtypes(exclude=['object'])
    return df.loc[:, df.nunique() > 1]

# ======= Loại bỏ đặc trưng tương quan cao =======
def remove_highly_correlated_features(df, threshold=0.7):
    corr_matrix = df.corr().abs()
    upper = corr_matrix.where(
        np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
    )
    to_drop = [col for col in upper.columns if any(upper[col] > threshold)]
    return df.drop(columns=to_drop)
