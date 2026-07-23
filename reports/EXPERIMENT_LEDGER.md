# Experiment ledger

This ledger records completed Colab experiments exactly as reported by their
per-fold terminal output and aggregate summaries. It is a provenance record,
not a claim of final performance.

## v3 five-fold run: prototype alignment with combined weak guidance (2026-07-22)

Protocol: `sensory-v3` (core taste = sweet, bitter, umami; sour and salty are
low-shot endpoints outside training and the headline metric). Scaffold-disjoint
folds 0–4 with validation = (test+1) % 5, seed 42, maximum 30 epochs, early
stopping patience 6, LoRA on the last 4 layers with rank 4 (127,122 trainable
parameters). Losses: prototype NCE 0.05, strong label-set alignment 0.05
(τ=0.07), weak FlavorDB BCE 0.02, weak alignment 0.01 (τ=0.5). Training-split
`pos_weight` capped at 10 on the curated odor/taste BCE. Per-label decision
thresholds were selected on each fold's validation split and locked into the
checkpoint before the single test evaluation.

### Held-out test metrics (validation-locked thresholds)

| Metric | Mean ± SD | Per fold |
| --- | ---: | --- |
| Combined score | 0.6258 ± 0.0247 | 0.5852, 0.6326, 0.6302, 0.6284, 0.6527 |
| Odor macro-F1 (12 families) | 0.4943 ± 0.0097 | 0.4844, 0.4869, 0.5083, 0.4994, 0.4925 |
| Taste macro-F1 (3 core labels) | 0.7573 ± 0.0465 | 0.6859, 0.7783, 0.7520, 0.7574, 0.8130 |

The pre-declared, adequately supported odor families (fruity, floral, green,
woody, fatty, sulfurous, sweet_aromatic) all reach ≥0.53 per-fold mean F1
(combined macro ≈ 0.62); the long-tail families (spicy, animalic, phenolic,
nutty) remain unreliable. Taste per label: bitter 0.8152 ± 0.0563, sweet
0.7776 ± 0.0604, umami 0.6792 ± 0.0340.

### Pair retrieval probes (test projections, within-modality)

| Probe | R@1 | R@5 | MRR | Queries per fold |
| --- | ---: | ---: | ---: | --- |
| taste_profile | 0.6000 ± 0.0608 | 0.8963 ± 0.0374 | 0.7349 ± 0.0324 | 21–38 |
| odor_profile | 0.1131 ± 0.0451 | 0.2323 ± 0.1318 | 0.1903 ± 0.0633 | 16–26 |

### Reading the v3 result

- Fold 0 is a low outlier (0.5852); folds 1–4 lie within 0.6284–0.6527.
  Single-fold reporting would have misestimated this model by ±0.03.
- Nutty swings 0.087 → 0.585 across folds (SD 0.19): rare-label single-fold
  numbers are meaningless, which is why only the pre-declared common families
  support fold-level claims.
- Validation-locked thresholds helped on 4 of 5 folds (up to +0.042 on fold
  2) and cost 0.003 on fold 1: a small net positive, retained for protocol
  honesty rather than for gain.
- Bitter's F1-optimal threshold is 0.1–0.2 and umami's is 0.74–0.98 on every
  fold: a stable calibration pathology that deserves a diagnostic note, not a
  footnote.
- The retrieval probes measure within-modality profile organisation only. The
  134-pair cross-sensory alignment question is **not** answered by this run,
  and taste R@5 = 0.90 must not be read as cross-modal evidence.
- The v2 entries below are not comparable with this run: the task definition
  (sour removed from the core task), the alignment architecture (label
  prototypes instead of same-molecule projection pairs), and the evaluation
  protocol (pos_weight, locked thresholds) all changed at once.

Artifacts: `outputs/v3_prototype_d/fold{0..4}_metrics.json` and
`reports/v3_prototype_d_5fold/summary.{json,md}`. Re-running
`scripts/aggregate_cross_sensory.py` on the archived fold files reproduces
`summary.json` bit-identically.

## 2D control: Morgan fingerprint + logistic regression (2026-07-23)

Protocol: identical scaffold folds, masked targets, and validation-locked
threshold protocol as the v3 five-fold run above. Morgan fingerprints
(radius 2, 2048 bits) with per-label balanced logistic regression
(liblinear, C=1.0), each label fit on its observed training rows. No
retrieval probes: the fingerprint space is not the learned projection.

| Metric | 2D control (Mean ± SD) | Uni-Mol LoRA 3D (Mean ± SD) |
| --- | ---: | ---: |
| Odor macro-F1 | 0.4849 ± 0.0069 | 0.4943 ± 0.0097 |
| Taste macro-F1 | 0.7855 ± 0.0365 | 0.7573 ± 0.0465 |
| Combined score | 0.6352 ± 0.0182 | 0.6258 ± 0.0247 |

### Reading the 2D-vs-3D comparison

- **Taste: the 2D control wins every fold** (mean +0.028; paired Wilcoxon
  p=0.0625, the minimum attainable at n=5) and every label (sweet +0.058,
  bitter +0.020, umami +0.007). Consistent with the broader evidence that
  fingerprint features capture sweet/bitter pharmacophore patterns well.
- **Odor: the 3D model leads on 4 of 5 folds** (mean +0.009; p=0.1875) and
  9 of 12 labels, with the largest margins on sulfurous (+0.043) and woody
  (+0.026). Suggestive but underpowered at n=5.
- **Combined score is a statistical tie** (p=0.1875). The honest headline:
  no demonstrated overall advantage for the 3D foundation model over a free
  fingerprint on this benchmark — mirroring the field-wide picture of
  arXiv:2508.06199 at sensory scale.
- Caveats: n=5 bounds attainable power (min p=0.0625); logistic regression
  is the weaker fingerprint classifier (a fingerprint + random forest
  control remains); the LoRA run may be undertrained (validation still
  rising at epoch 30 on folds 1–2), so a longer-training ablation is the
  fair rematch before conceding odor.

Artifacts: `outputs/v3_fingerprint_lr/fold{0..4}_metrics.json` and
`reports/v3_fingerprint_lr_5fold/summary.{json,md}` (bit-identical
re-aggregation verified).

## Historical v2 fold 0 ablations (`sensory-v2`, not comparable to v3)

All entries in this section used the historical `sensory-v2` four-label taste
task (`sweet`, `bitter`, `sour`, `umami`), Fold 0 as test, Fold 1 as
validation, scaffold-disjoint folds, seed 42, maximum 30 epochs, and early
stopping patience 6.

| ID | LoRA | Weak BCE | Weak InfoNCE | Weak temperature | Test odor F1 | Test taste F1 | Test score |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Baseline | last 4 layers, rank 4 | 0 | 0 | — | 0.3867 | 0.5375 | 0.4621 |
| A: heavy weak guidance | last 8 layers, rank 4 | 0.15 | 0.01 | 0.2 | 0.3638 | 0.5338 | 0.4488 |
| C: weak contrastive only | last 4 layers, rank 4 | 0 | 0.02 | 0.5 | 0.4012 | 0.5381 | 0.4697 |
| D: combined weak guidance | last 4 layers, rank 4 | 0.02 | 0.01 | 0.5 | **0.4025** | **0.5401** | **0.4713** |

All weak-guidance runs retained the strong-pair objective
(`contrastive_weight=0.05`, strong temperature `0.07`) and used two strong
plus two weak-only paired molecules per batch.

### Reading the v2 result

- Heavy guidance with LoRA-8 regressed, so depth and guidance strength should
  not be increased together without an ablation.
- Weak high-temperature InfoNCE improved the held-out odor macro-F1 by 0.0145
  and the combined score by 0.0076 over the v2 baseline.
- Adding low-weight weak BCE produced the best observed Fold 0 score: +0.0158
  odor F1, +0.0027 taste F1, and +0.0092 combined score relative to baseline.
- The C-to-D difference is only 0.0017 combined score. One fold cannot
  establish that weak BCE adds a reliable incremental benefit.

### Limits of the v2 entries

These are single-fold results; they do not estimate split variance. Sour had
too little curated support for a stable Fold 0 result and is now grouped with
salty as a low-shot endpoint under `sensory-v3`. None of these v2 numbers may
be compared directly with the v3 run above.
