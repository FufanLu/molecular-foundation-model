# Experiment ledger

This ledger records completed Colab experiments exactly as reported by their
per-fold terminal output. It is a provenance record, not a claim of final
performance. All entries below used the historical `sensory-v2` four-label
taste task (`sweet`, `bitter`, `sour`, `umami`), Fold 0 as test, Fold 1 as
validation, scaffold-disjoint folds, seed 42, maximum 30 epochs, and early
stopping patience 6.

## Fold 0 ablations

| ID | LoRA | Weak BCE | Weak InfoNCE | Weak temperature | Test odor F1 | Test taste F1 | Test score |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Baseline | last 4 layers, rank 4 | 0 | 0 | — | 0.3867 | 0.5375 | 0.4621 |
| A: heavy weak guidance | last 8 layers, rank 4 | 0.15 | 0.01 | 0.2 | 0.3638 | 0.5338 | 0.4488 |
| C: weak contrastive only | last 4 layers, rank 4 | 0 | 0.02 | 0.5 | 0.4012 | 0.5381 | 0.4697 |
| D: combined weak guidance | last 4 layers, rank 4 | 0.02 | 0.01 | 0.5 | **0.4025** | **0.5401** | **0.4713** |

All weak-guidance runs retained the strong-pair objective
(`contrastive_weight=0.05`, strong temperature `0.07`) and used two strong
plus two weak-only paired molecules per batch.

## Reading the result

- Heavy guidance with LoRA-8 regressed, so depth and guidance strength should
  not be increased together without an ablation.
- Weak high-temperature InfoNCE improved the held-out odor macro-F1 by 0.0145
  and the combined score by 0.0076 over the v2 baseline.
- Adding low-weight weak BCE produced the best observed Fold 0 score: +0.0158
  odor F1, +0.0027 taste F1, and +0.0092 combined score relative to baseline.
- The C-to-D difference is only 0.0017 combined score. One fold cannot
  establish that weak BCE adds a reliable incremental benefit.

## Limits and next comparison

These are single-fold results; they do not estimate split variance. Sour had
too little curated support for a stable Fold 0 result and is now grouped with
salty as a low-shot endpoint under `sensory-v3`. Therefore none of these v2
numbers may be compared directly with a future v3 run.

When v3 training resumes, run the same configuration across all five test
folds, then aggregate the five `fold*_metrics.json` files:

```bash
python scripts/aggregate_cross_sensory.py \
  --metrics outputs/v3_d/fold0_metrics.json outputs/v3_d/fold1_metrics.json \
            outputs/v3_d/fold2_metrics.json outputs/v3_d/fold3_metrics.json \
            outputs/v3_d/fold4_metrics.json \
  --output-dir reports/v3_d_5fold
```

The aggregator rejects duplicate folds, incompatible alignment settings, or
incompatible core/low-shot task definitions.
