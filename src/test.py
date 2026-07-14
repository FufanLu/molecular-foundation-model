from dataset.load_leffingwell import load_leffingwell
from preprocessing.clean_smiles import clean_smiles
from preprocessing.generate_conformer import generate_conformers

df = load_leffingwell()
df = clean_smiles(df)
generate_conformers(df)