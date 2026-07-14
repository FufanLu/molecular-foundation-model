from pathlib import Path
import pandas as pd

DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "raw" / "leffingwell"

def load_leffingwell():
    csv_path = DATA_DIR / "leffingwell_merged.csv"
    df = pd.read_csv(csv_path)
    df = df.rename(
        columns={
            "IsomericSMILES": "smiles",
            "name": "compound",
            "Labels": "labels",
        }
    )
    return df[["compound", "smiles", "labels"]].copy()

