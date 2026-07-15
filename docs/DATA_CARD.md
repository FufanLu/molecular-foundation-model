# Data card

## Purpose

This project prepares a molecule-level corpus for exploratory odor/taste
learning. It is not a clinical, toxicological, or sensory-safety dataset.

## Source inputs

| Source | Role in this project |
| --- | --- |
| Leffingwell | Odor descriptor evidence and molecular structures. |
| GoodScents | Odor descriptor evidence and molecular structures. |
| FlavorDB | Odor descriptors plus **weak** flavor wording. |
| ChemTastesDB | Curated basic-taste labels and molecular structures. |

The raw files are stored under `data/raw/leffingwell/` for the current
reproducible snapshot. Every downstream user is responsible for complying with
the original source terms, attribution requirements, and any restrictions on
redistribution or commercial use.

## Processing

1. Validate and canonicalise source SMILES with RDKit.
2. Use InChIKey when available to aggregate exact molecules across sources.
3. Preserve source names, source CIDs, names, raw descriptor text, and a
   mixture/salt flag.
4. Map odor terms to a documented 12-family multi-label ontology.
5. Keep FlavorDB taste wording in `taste_weak_labels`.
6. Use ChemTastesDB `Class taste` and documented multitaste records for
   `taste_strong_labels`.
7. Create deterministic scaffold-disjoint, multilabel-balanced folds.

## Current audited snapshot

| Quantity | Value |
| --- | ---: |
| Source records | 37,821 |
| Valid structures | 37,818 |
| Unique canonical molecules | 31,971 |
| Curated taste-labelled molecules | 3,256 |
| Exact odor + curated-taste pairs | 134 |
| Salty molecules | 47 |
| Dot-disconnected mixtures/salts | 3,164 |
| CID-to-multiple-structure conflicts | 0 |

Curated labels currently include bitter (1,719), sweet (1,353), umami (292),
sour (100), and salty (47). These are multi-label counts, not mutually
exclusive classes.

## Known limitations

- Descriptor absence in a source is not universally equivalent to a true
  sensory negative; label completeness varies by source.
- Odor descriptors are semantically broad and can overlap.
- ChemTastesDB adds stronger basic-taste evidence, but sour and salty remain
  too small for reliable conventional classifier claims.
- Canonical structure matching does not account for concentration, solvent,
  formulation, stereochemical uncertainty in a source, or experimental setup.
- The dataset should not be used to infer human safety, exposure limits,
  toxicity, or a person's sensory response.

## Recommended reporting

Report source version, ontology version, fold assignment, threshold-selection
procedure, and per-label support. Keep weak FlavorDB wording separate from
curated taste labels in both training and claims.
