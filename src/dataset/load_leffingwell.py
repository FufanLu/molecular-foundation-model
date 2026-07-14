from pathlib import Path
import ast
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
    df["labels"] = df["labels"].apply(_parse_labels)
    return df[["compound", "smiles", "labels"]].copy()


def _parse_labels(x):
    if isinstance(x, list):
        return x
    if isinstance(x, str):
        try:
            return ast.literal_eval(x)
        except (ValueError, SyntaxError):
            return []
    return []


