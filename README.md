# Cross-Sensory Molecular Foundation Model

An auditable, reproducible baseline for learning molecular representations
across **odor** and **taste**. The project combines Uni-Mol LoRA adaptation,
curated basic-taste supervision, mapped odor families, and molecule-level
sensory-label prototype alignment.

The central design choice is to keep evidence quality explicit:

- **Curated taste labels** from ChemTastesDB define the supervised core task:
  sweet, bitter, and umami. Sour and salty are retained as low-shot endpoints,
  not folded into training or the headline metric.
- **FlavorDB taste words** are retained as weak descriptors only; they are not
  silently treated as physiological basic-taste ground truth.
- Odor and taste records are paired only after RDKit canonicalisation and
  molecule-level provenance checks.

## Current v3 result (five scaffold folds, 2026-07-22)

The first complete `sensory-v3` run used the prototype-alignment objective
with combined weak guidance. Validation-locked per-label thresholds were
selected inside each fold before the single test evaluation.

| Held-out test metric (n=5 folds) | Mean ± SD |
| --- | ---: |
| Odor macro-F1, 12 mapped families | 0.4943 ± 0.0097 |
| Curated taste macro-F1, 3 core labels | 0.7573 ± 0.0465 |
| Combined score | 0.6258 ± 0.0247 |

The seven pre-declared, adequately supported odor families reach a combined
macro F1 of ≈0.62; long-tail families (spicy, animalic, phenolic, nutty)
remain unreliable, with nutty swinging 0.087→0.585 across folds. See the
[experiment ledger](reports/EXPERIMENT_LEDGER.md) for the full per-fold table,
retrieval probes, and reading notes.

## Historical reference run

The following Fold 0 result used the earlier `sensory-v2` four-label taste
head. It is retained for provenance and is **not** directly comparable to new
`sensory-v3` three-label runs.

| Held-out test metric | Score |
| --- | ---: |
| Odor macro-F1, 12 mapped families | 0.3867 |
| Curated taste macro-F1, 4 labels (v2) | 0.5375 |
| Mean task score | 0.4621 |

It establishes useful signals for common labels while exposing poor scaffold
transfer for rare labels. See the [Fold 0 report](reports/cross_sensory_fold0_baseline.md)
for the historical class-level results and limitations.

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
- `paired`: exact molecules with odor labels and a curated core-taste label.
- `is_mixture`: dot-disconnected salts or mixtures; excluded from the main
  Uni-Mol benchmark by default and retained for low-shot salt analysis.

The current model maps each 3D molecular embedding into one shared sensory
space. Odor and taste labels are learned as separate prototypes in that space:
molecules align to their observed label prototypes, while strong/weak evidence
aligns odor-label sets to taste-label sets. A separate weak-flavor head keeps
FlavorDB wording distinct from curated taste supervision.

The current audited corpus has 37,821 source records, 31,971 unique canonical
molecules, 3,256 curated taste-labelled molecules, 134 exact odor--taste
pairs, and 47 salty molecules. Read [the data card](docs/DATA_CARD.md) before
interpreting results.

## Experiment tracking

The [experiment ledger](reports/EXPERIMENT_LEDGER.md) preserves the completed
historical v2 Fold 0 ablations. Future v3 runs should be summarised only after
all available test folds are complete:

```bash
python scripts/aggregate_cross_sensory.py \
  --metrics outputs/v3_prototype_d/fold*_metrics.json \
  --output-dir reports/v3_prototype_d_5fold
```

The aggregator writes a machine-readable `summary.json` and a Markdown table
with held-out mean ± standard deviation. It rejects mixed task definitions,
alignment settings, and duplicate test folds.

## Quick start: Colab

The supported execution environment is a Colab GPU runtime. Open
[the Colab notebook](notebooks/train_cross_sensory_colab.ipynb), then:

1. Clone or upload this current repository revision.
2. Provide the four source files directly under `data/raw/`.
3. Run the preparation cell to create the `sensory-v3` processed dataset.
   Re-run this step after pulling this revision; `sensory-v2` files are
   intentionally rejected by training.
4. Run one scaffold fold (`--folds 0`) before launching all five folds.

The notebook intentionally uses `multi_process=False` for Uni-Mol input
generation; this avoids CUDA/fork deadlocks in Colab.

For a shell-based Colab run after preparation:

```bash
PYTHONPATH=. python scripts/train_cross_sensory.py \
  --data data/processed/sensory/molecules.parquet \
  --output-dir outputs/v3_prototype_d \
  --folds 0,1,2,3,4 \
  --epochs 30 --patience 6 \
  --batch-size 16 --lora-layers 4 --lora-rank 4 \
  --paired-per-batch 2 --weak-paired-per-batch 2 \
  --prototype-weight 0.05 --contrastive-weight 0.05 \
  --weak-taste-weight 0.02 --weak-contrastive-weight 0.01 \
  --weak-temperature 0.5
```

Training behaviour worth knowing:

- Uni-Mol inputs are featurised once per SMILES list and cached as
  `data/processed/sensory/unimol_features_<digest>.pkl`; every fold reuses
  the cache, so all five folds share identical conformers. Use
  `--refresh-features` to force recomputation.
- `--folds` validates on fold `(test + 1) % 5` and skips any fold whose
  metrics JSON already exists, so an interrupted Colab run resumes cheaply.
- Per-label decision thresholds are selected on each fold's validation split
  and locked into the checkpoint; the test split is evaluated once with them.
  A fixed-0.5 reference is kept in `test_at_fixed_threshold`.
- The masked BCE uses training-split `pos_weight` (capped by
  `--pos-weight-cap`, default 10) for the curated odor/taste tasks; the weak
  FlavorDB head stays unweighted.
- Each metrics JSON also reports retrieval probes (Recall@1/5, MRR): whether
  paired molecules' projections recover identical sensory label sets.

## Repository layout

```text
src/
  dataset/sensory.py          # ingestion, standardisation, audit, grouped folds
  sensory/                    # LoRA, shared encoder, prototype-alignment losses,
                              # thresholds/pos_weight/retrieval metrics
scripts/train_cross_sensory.py
notebooks/train_cross_sensory_colab.ipynb
reports/                      # immutable experiment reports
docs/DATA_CARD.md             # provenance, label contract, limitations
```

## Scope and limitations

- Odor descriptors are expert/source annotations, not controlled perceptual
  panel measurements.
- The 134 strong molecule pairs are sufficient for an exploratory label-set
  alignment objective, not for a standalone claim of cross-modal alignment.
- Sour and salty are low-resource endpoints. Both are excluded from the main
  training objective and core taste macro-F1, and must be evaluated separately.
- A final study requires all five folds, validation-only threshold selection,
  odor-only/multitask/contrastive ablations, and independent validation before
  making strong biological or perceptual claims.

## Data and license

The code is released under the [MIT License](LICENSE). Source datasets retain
their original provenance and terms; see [the data card](docs/DATA_CARD.md).
Do not assume that the code license grants redistribution rights for every raw
source record.
