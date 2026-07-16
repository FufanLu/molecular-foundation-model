# Cross-sensory Uni-Mol baseline — Fold 0 (historical `sensory-v2`)

**Run:** shared Uni-Mol encoder + last-4-layer LoRA + odor/taste heads +
pair-aware InfoNCE (`contrastive_weight=0.05`)
**Split:** scaffold-disjoint; fold 0 test, fold 1 validation
**Training:** 30 epochs maximum, best validation checkpoint retained

> This historical run trained a four-label taste head. The current
> `sensory-v3` protocol uses sweet, bitter, and umami as the core taste task;
> sour and salty are separate low-shot endpoints. Do not compare its taste
> macro-F1 directly with a v3 run.

## Headline result

| Split | Odor macro-F1 | Taste macro-F1 | Mean task score |
| --- | ---: | ---: | ---: |
| Best validation checkpoint | 0.4245 | 0.6896 | 0.5571 |
| Held-out scaffold test | 0.3867 | 0.5375 | 0.4621 |

The test score is the reportable number for this fold.  The validation--test
gap shows that generalising to unseen molecular scaffolds remains difficult,
especially for rare odor and taste labels.

## Odor-family results

| Odor family | Validation F1 | Test F1 | Reading |
| --- | ---: | ---: | --- |
| fruity | 0.7655 | 0.6529 | Strongest common odor signal. |
| floral | 0.4804 | 0.5673 | Generalises reasonably. |
| green | 0.6300 | 0.4889 | Moderate scaffold transfer. |
| woody | 0.5273 | 0.2536 | Marked transfer drop. |
| fatty | 0.6300 | 0.5191 | Usable common-label result. |
| sulfurous | 0.6623 | 0.7811 | Strongest test generalisation. |
| spicy | 0.0833 | 0.2476 | Sparse/noisy; unstable. |
| sweet_aromatic | 0.4571 | 0.6075 | Moderate result. |
| nutty | 0.2074 | 0.0381 | Collapsed on held-out scaffolds. |
| animalic | 0.2000 | 0.1887 | Insufficiently learned. |
| phenolic | 0.0833 | 0.0248 | Collapsed on held-out scaffolds. |
| aldehydic | 0.3673 | 0.2705 | Weak transfer. |
| **Macro average** | **0.4245** | **0.3867** | Long-tail labels dominate the macro penalty. |

## Curated taste results

| Taste | Validation F1 | Test F1 | Reading |
| --- | ---: | ---: | --- |
| sweet | 0.8379 | 0.7209 | Strong, transferable signal. |
| bitter | 0.8475 | 0.7186 | Strong, transferable signal. |
| sour | 0.4000 | 0.0000 | Too sparse for the current fixed-threshold setup. |
| umami | 0.6731 | 0.7103 | Strong test transfer. |
| **Macro average** | **0.6896** | **0.5375** | Mainly reduced by sour. |

Under the current v3 protocol, both `sour` and `salty` are separate low-shot
probes and excluded from the core taste training objective and macro-F1.

## What this run establishes

- The model has learned useful structure-to-sensory signals: fruity, fatty,
  sulfurous, sweet, bitter, and umami all have meaningful held-out F1.
- The current full 12-odor macro-F1 is not yet a robust headline metric:
  nutty and phenolic almost entirely fail on unseen scaffolds.
- The cross-sensory objective is feasible, but 134 exact curated odor--taste
  pairs are too few to claim that contrastive alignment caused the gains.
- The current checkpoint is a valid **multitask baseline**, not evidence that
  the contrastive component improves performance.

## Next experiment set

1. Add training-split `pos_weight` for each odor label and select one threshold
   per label on validation data; evaluate the locked thresholds once on test.
2. Compare three otherwise identical runs: odor-only, odor+taste with
   `contrastive_weight=0`, and the current `contrastive_weight=0.05` setup.
3. Report a pre-declared common-label subset separately from the full 12-label
   exploratory odor result; retain the full result for transparency.
4. Add pair retrieval metrics (Recall@1, Recall@5, MRR) before making any
   cross-modal alignment claim.

## Reproducibility note

This report is for one held-out fold only.  Final claims require all five
scaffold folds, with thresholds selected inside each fold's validation split.
