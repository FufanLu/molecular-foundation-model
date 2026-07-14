# Molecule Foundation Model

FlavorDB → SMILES → RDKit → 3D → Uni-Mol → Embedding → Similarity

## Phase 1: Molecular Embedding Extraction

Extract molecular embeddings from FlavorDB2 using Uni-Mol pre-trained model.

## Structure

```
src/
├── dataset/          # Data loading
├── preprocessing/    # SMILES cleaning & 3D conformer generation
├── encoder/          # Uni-Mol embedding extraction
└── evaluation/       # Embedding quality validation

data/
└── raw/
    └── flavordb/     # FlavorDB2 raw data

notebooks/            # Demo notebooks
```

## Pipeline

1. Load FlavorDB2
2. Clean SMILES (dedup, RDKit validation)
3. Generate 3D conformers (ETKDG + UFF)
4. Extract Uni-Mol embeddings
5. Validate via molecular similarity
