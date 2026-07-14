from pathlib import Path
import pickle
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem

CONFORMER_DIR = Path(__file__).resolve().parents[2] / "data" / "processed" / "conformers"


def smiles_to_3d(smiles: str):
    mol = Chem.MolFromSmiles(smiles)
    mol = Chem.AddHs(mol)
    AllChem.EmbedMolecule(mol, randomSeed=42)
    AllChem.UFFOptimizeMolecule(mol)
    mol = Chem.RemoveHs(mol)
    return mol


def generate_conformers(df: pd.DataFrame):
    CONFORMER_DIR.mkdir(parents=True, exist_ok=True)

    pkl_path = CONFORMER_DIR / "conformers.pkl"
    mols = {}

    for i, row in df.iterrows():
        compound = row["compound"]
        smiles = row["smiles"]
        try:
            mol = smiles_to_3d(smiles)
            mols[compound] = mol
        except Exception as e:
            print(f"skip {compound}: {e}")

    with open(pkl_path, "wb") as f:
        pickle.dump(mols, f)

    print(f"{len(mols)} / {len(df)} conformers saved to {pkl_path}")
    return mols
