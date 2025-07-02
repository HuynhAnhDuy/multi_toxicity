import pandas as pd
import numpy as np
from rdkit import Chem

def canonical_smiles(df, smiles_col):
    """
    Convert SMILES to canonical form using RDKit.
    """
    df['canonical_smiles'] = df[smiles_col].apply(
        lambda x: Chem.MolToSmiles(Chem.MolFromSmiles(x), isomericSmiles=True)
    )
    return df


def has_carbon_atoms(smiles):
    """
    Check if a molecule has at least one carbon atom (C, atomic number 6).
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol:
        return any(atom.GetAtomicNum() == 6 for atom in mol.GetAtoms())
    return False


def remove_inorganic(df, smiles_col):
    """
    Remove molecules that do not contain carbon atoms.
    """
    df['has_carbon'] = df[smiles_col].apply(has_carbon_atoms)
    df = df[df['has_carbon']]
    df = df.drop(columns=['has_carbon'])
    return df


def remove_mixtures(df, smiles_col):
    """
    Remove mixtures (molecules with multiple components, e.g. "C.CC").
    """
    df['is_mixture'] = df[smiles_col].apply(lambda x: '.' in x)
    df = df[~df['is_mixture']]
    df = df.drop(columns=['is_mixture'])
    return df


def remove_constant_string_des(df):
    """
    Remove string-type and constant-value descriptor columns.
    """
    df = df.select_dtypes(exclude=['object'])  # remove string columns
    for col in df.columns:
        if df[col].nunique() <= 1:
            df = df.drop(columns=[col])
    return df


def remove_highly_correlated_features(df, threshold=0.7):
    """
    Remove highly correlated features above the given threshold.
    """
    corr_matrix = df.corr().abs()
    upper = corr_matrix.where(
        pd.DataFrame(
            np.triu(np.ones(corr_matrix.shape), k=1).astype(bool),
            index=corr_matrix.index,
            columns=corr_matrix.columns
        )
    )
    to_drop = [col for col in upper.columns if any(upper[col] > threshold)]
    return df.drop(columns=to_drop)
