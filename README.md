# Cross-Sensory Molecular Foundation Model

An auditable, reproducible baseline for learning molecular representations
across **odor** and **taste**. The project combines Uni-Mol LoRA adaptation,
curated basic-taste supervision, mapped odor families, and molecule-level
cross-sensory contrastive learning.

The central design choice is to keep evidence quality explicit:

- **Curated taste labels** come from ChemTastesDB and define the supervised
  sweet, bitter, sour, and umami task.
- **FlavorDB taste words** are retained as weak descriptors only; they are not
  silently treated as physiological basic-taste ground truth.
- Odor and taste records are paired only after RDKit canonicalisation and
  molecule-level provenance checks.

## Current baseline

The first scaffold-disjoint Fold 0 baseline uses a shared Uni-Mol encoder,
last-four-layer LoRA, separate odor/taste heads, and pair-aware InfoNCE.

| Held-out test metric | Score |
| --- | ---: |
| Odor macro-F1, 12 mapped families | 0.3867 |
| Curated taste macro-F1, 4 labels | 0.5375 |
| Mean task score | 0.4621 |

This is a baseline, not a final cross-modal claim. It establishes useful
signals for common labels while exposing poor scaffold transfer for rare
labels. See the [Fold 0 report](reports/cross_sensory_fold0_baseline.md) for
all class-level results and limitations.

## Dataset contract

After preparation, `data/processed/sensory/` contains:

| Artifact | Purpose |
| --- | --- |
| `source_records.parquet` | Every source row, raw label evidence, structure status, and provenance. |
| `molecules.parquet` | Canonical molecule-level aggregation, label tiers, mixture flag, and frozen folds. |
| `audit.json` | Counts, integrity checks, and per-fold label coverage. |

`molecules.parquet` separates the following fields:

- `odor_labels`: mapped multi-label odor families.
- `taste_strong_labels` / `taste_labels`: ChemTastesDB curated taste labels.
- `taste_weak_labels`: FlavorDB descriptor words; weak evidence only.
- `paired`: exact molecules with odor labels and a curated main-taste label.
- `is_mixture`: dot-disconnected salts or mixtures; excluded from the main
  Uni-Mol benchmark by default and retained for low-shot salt analysis.

The next training configuration keeps a separate weak-flavor head: strong
ChemTastesDB pairs use low-temperature InfoNCE, while weak-only FlavorDB pairs
use high-temperature, low-weight guidance. This is evaluated as an ablation;
the current Fold 0 report remains the strong-pair baseline.

The current audited corpus has 37,821 source records, 31,971 unique canonical
molecules, 3,256 curated taste-labelled molecules, 134 exact odor--taste
pairs, and 47 salty molecules. Read [the data card](docs/DATA_CARD.md) before
interpreting results.

## Quick start: Colab

The supported execution environment is a Colab GPU runtime. Open
[the Colab notebook](notebooks/train_cross_sensory_colab.ipynb), then:

1. Clone or upload this current repository revision.
2. Provide the four source files under `data/raw/leffingwell/`.
3. Run the preparation cell to create the processed sensory dataset.
4. Run one scaffold fold before launching the remaining four folds.

The notebook intentionally uses `multi_process=False` for Uni-Mol input
generation; this avoids CUDA/fork deadlocks in Colab.

For a shell-based Colab run after preparation:

```bash
PYTHONPATH=. python scripts/train_cross_sensory.py \
  --data data/processed/sensory/molecules.parquet \
  --output-dir outputs/cross_sensory_weak_lora4_w002 \
  --test-fold 0 --val-fold 1 \
  --epochs 30 --patience 6 \
  --batch-size 16 --lora-layers 4 --lora-rank 4 \
  --paired-per-batch 2 --weak-paired-per-batch 2 \
  --contrastive-weight 0.05 --weak-taste-weight 0.02 \
  --weak-contrastive-weight 0 --weak-temperature 0.2
```

## Repository layout

```text
src/
  dataset/sensory.py          # ingestion, standardisation, audit, grouped folds
  sensory/                    # LoRA, shared encoder, supervised + contrastive losses
scripts/train_cross_sensory.py
notebooks/train_cross_sensory_colab.ipynb
reports/                      # immutable experiment reports
docs/DATA_CARD.md             # provenance, label contract, limitations
```

## Scope and limitations

- Odor descriptors are expert/source annotations, not controlled perceptual
  panel measurements.
- The 134 strong molecule pairs are sufficient for an exploratory contrastive
  objective, not for a standalone claim of cross-modal alignment.
- Sour and salty are low-resource settings. Salty is excluded from the main
  taste macro-F1 and should be evaluated separately.
- A final study requires all five folds, validation-only threshold selection,
  odor-only/multitask/contrastive ablations, and independent validation before
  making strong biological or perceptual claims.

## Data and license

The code is released under the [MIT License](LICENSE). Source datasets retain
their original provenance and terms; see [the data card](docs/DATA_CARD.md).
Do not assume that the code license grants redistribution rights for every raw
source record.
