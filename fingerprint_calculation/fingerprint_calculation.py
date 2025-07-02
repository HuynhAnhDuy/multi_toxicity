#importing libraries
import pandas as pd
import numpy as np
from padelpy import padeldescriptor
from rdkit.Chem import AllChem as Chem
from rdkit import DataStructs
import os
from glob import glob

def canonical_smiles(df, smiles_column):
    df['canonical_smiles'] = df[smiles_column].apply(lambda x: Chem.MolToSmiles(Chem.MolFromSmiles(x)))
    return df
def compute_fps(df, name, path="train"):
    xml_files = glob(os.path.join(name,"*.xml"))
    xml_files.sort()
    FP_list = [
    'AP2DC','KRFPC','KRFP','PubChem','SubFPC','SubFP']
    fp = dict(zip(FP_list, xml_files))
    print(fp)
    df['canonical_smiles'].to_csv(os.path.join(name, 'smiles.smi'), sep='\t', index=False, header=False)
    #Calculate fingerprints
    for i in FP_list:
        padeldescriptor(mol_dir=os.path.join(name, 'smiles.smi'),
                    d_file=os.path.join(name, i+'.csv'),
                    descriptortypes= fp[i],
                    retainorder=True, 
                    removesalt=True,
                    threads=2,
                    detectaromaticity=True,
                    standardizetautomers=True,
                    standardizenitro=True,
                    fingerprints=True
                    )
        Fingerprint = pd.read_csv(os.path.join(name, i+'.csv')).set_index(df.index)
        Fingerprint = Fingerprint.drop('Name', axis=1)
        Fingerprint.to_csv(os.path.join(name,path, i+'.csv'))
        print(i+'.csv', 'done')
    #load at pc
    fp_at = pd.read_csv(os.path.join(name,path,'AP2D.csv'    ) , index_col=0)
    fp_es = pd.read_csv(os.path.join(name,path,'EState.csv'  ) , index_col=0)
    fp_ke = pd.read_csv(os.path.join(name,path,'KRFP.csv'    ) , index_col=0)
    fp_pc = pd.read_csv(os.path.join(name,path,'PubChem.csv' ) , index_col=0)
    fp_ss = pd.read_csv(os.path.join(name,path,'SubFP.csv'   ) , index_col=0)
    fp_cd = pd.read_csv(os.path.join(name,path,'CDKGraph.csv') , index_col=0)
    fp_cn = pd.read_csv(os.path.join(name,path,'CDK.csv'     ) , index_col=0)
    fp_kc = pd.read_csv(os.path.join(name,path,'KRFPC.csv'   ) , index_col=0)
    fp_ce = pd.read_csv(os.path.join(name,path,'CDKExt.csv'  ) , index_col=0)
    fp_sc = pd.read_csv(os.path.join(name,path,'SubFPC.csv'  ) , index_col=0)
    fp_ac = pd.read_csv(os.path.join(name,path,'AP2DC.csv'   ) , index_col=0)
    fp_ma = pd.read_csv(os.path.join(name,path,'MACCS.csv'   ) , index_col=0)
    
    fp_at.to_csv(os.path.join(name, path,'AP2D.csv'    ))
    fp_es.to_csv(os.path.join(name, path,'EState.csv'  ))
    fp_ke.to_csv(os.path.join(name, path,'KRFP.csv'    ))
    fp_pc.to_csv(os.path.join(name, path,'PubChem.csv' ))
    fp_ss.to_csv(os.path.join(name, path,'SubFP.csv'   ))
    fp_cd.to_csv(os.path.join(name, path,'CDKGraph.csv'))
    fp_cn.to_csv(os.path.join(name, path,'CDK.csv'     ))
    fp_kc.to_csv(os.path.join(name, path,'KRFPC.csv'   ))
    fp_ce.to_csv(os.path.join(name, path,'CDKExt.csv'  ))
    fp_sc.to_csv(os.path.join(name, path,'SubFPC.csv'  ))
    fp_ac.to_csv(os.path.join(name, path,'AP2DC.csv'   ))
    fp_ma.to_csv(os.path.join(name, path,'MACCS.csv'   ))
    return fp_at, fp_es, fp_ke, fp_pc, fp_ss, fp_cd, fp_cn, fp_kc, fp_ce, fp_sc, fp_ac, fp_ma


def main():
    name = 'Skin'
    df_train = pd.read_csv(os.path.join(name, "train", "x_train.csv"), index_col=0)
    fp_at, fp_es, fp_ke, fp_pc, fp_ss, fp_cd, fp_cn, fp_kc, fp_ce, fp_sc, fp_ac, fp_ma = compute_fps(df_train, name, path="train")
    df_test = pd.read_csv(os.path.join(name, "test", "x_test.csv"), index_col=0)
    fp_at, fp_es, fp_ke, fp_pc, fp_ss, fp_cd, fp_cn, fp_kc, fp_ce, fp_sc, fp_ac, fp_ma = compute_fps(df_test, name, path="test")
if __name__ == "__main__":
    main()
    