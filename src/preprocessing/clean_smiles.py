import pandas as pd
from rdkit import Chem


def clean_smiles(df: pd.DataFrame) -> pd.DataFrame:
    n_before = len(df)

    df = df.drop_duplicates(subset="smiles")
    df = df[df["smiles"].notna() & (df["smiles"] != "")]
    df = df[df["smiles"].apply(lambda s: Chem.MolFromSmiles(s) is not None)]

    n_after = len(df)
    print(f"{n_before} -> {n_after} valid molecules ({n_before - n_after} removed)")
    return df.reset_index(drop=True)
