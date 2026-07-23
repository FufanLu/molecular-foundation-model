# Cross-sensory fold summary

Folds: 0, 1, 2, 3, 4 (n=5)

## Protocol

- Core taste labels: sweet, bitter, umami
- Low-shot labels: sour, salty
- Alignment: `{"prototype_weight": 0.05, "strong_alignment_weight": 0.05, "strong_temperature": 0.07, "weak_contrastive_weight": 0.01, "weak_taste_weight": 0.02, "weak_temperature": 0.5}`

## Held-out test metrics

| Task | Metric | Mean ± SD | Per fold |
| --- | --- | ---: | --- |
| odor | aldehydic | 0.4327 ± 0.0464 | 0.3816, 0.5030, 0.4237, 0.4059, 0.4492 |
| odor | animalic | 0.2654 ± 0.0367 | 0.2128, 0.2456, 0.2920, 0.2727, 0.3038 |
| odor | fatty | 0.5999 ± 0.0268 | 0.5667, 0.6141, 0.5997, 0.6357, 0.5833 |
| odor | floral | 0.5509 ± 0.0686 | 0.6728, 0.5124, 0.5265, 0.5305, 0.5125 |
| odor | fruity | 0.7278 ± 0.0389 | 0.6603, 0.7565, 0.7473, 0.7303, 0.7443 |
| odor | green | 0.6473 ± 0.0298 | 0.5977, 0.6738, 0.6615, 0.6601, 0.6437 |
| odor | macro | 0.4943 ± 0.0097 | 0.4844, 0.4869, 0.5083, 0.4994, 0.4925 |
| odor | nutty | 0.3366 ± 0.1905 | 0.0870, 0.2188, 0.5849, 0.4076, 0.3848 |
| odor | phenolic | 0.2279 ± 0.0553 | 0.3202, 0.2105, 0.1957, 0.1798, 0.2336 |
| odor | spicy | 0.2983 ± 0.0828 | 0.4438, 0.2347, 0.2720, 0.2735, 0.2676 |
| odor | sulfurous | 0.7732 ± 0.0270 | 0.7958, 0.7702, 0.7806, 0.7911, 0.7282 |
| odor | sweet_aromatic | 0.5294 ± 0.0551 | 0.6217, 0.5377, 0.4899, 0.5072, 0.4906 |
| odor | woody | 0.5422 ± 0.0562 | 0.4527, 0.5656, 0.5263, 0.5983, 0.5679 |
| taste | bitter | 0.8152 ± 0.0563 | 0.7325, 0.8267, 0.8006, 0.8280, 0.8882 |
| taste | macro | 0.7573 ± 0.0465 | 0.6859, 0.7783, 0.7520, 0.7574, 0.8130 |
| taste | sweet | 0.7776 ± 0.0604 | 0.6736, 0.8178, 0.8016, 0.7776, 0.8174 |
| taste | umami | 0.6792 ± 0.0340 | 0.6517, 0.6905, 0.6538, 0.6667, 0.7333 |
| combined | score | 0.6258 ± 0.0247 | 0.5852, 0.6326, 0.6302, 0.6284, 0.6527 |

## Pair retrieval probes (test projections)

| Probe | Metric | Mean ± SD | Per fold |
| --- | --- | ---: | --- |
| odor_profile | mrr | 0.1903 ± 0.0633 | 0.1755, 0.1021, 0.2776, 0.1860, 0.2106 |
| odor_profile | recall@1 | 0.1131 ± 0.0451 | 0.0769, 0.0556, 0.1500, 0.1579, 0.1250 |
| odor_profile | recall@5 | 0.2323 ± 0.1318 | 0.1923, 0.1111, 0.4500, 0.1579, 0.2500 |
| odor_profile | queries | — | 26, 18, 20, 19, 16 |
| taste_profile | mrr | 0.7349 ± 0.0324 | 0.7593, 0.7557, 0.6982, 0.7606, 0.7008 |
| taste_profile | recall@1 | 0.6000 ± 0.0608 | 0.6579, 0.6364, 0.5455, 0.6364, 0.5238 |
| taste_profile | recall@5 | 0.8963 ± 0.0374 | 0.8947, 0.9545, 0.8636, 0.8636, 0.9048 |
| taste_profile | queries | — | 38, 22, 22, 22, 21 |

## Inputs

- `outputs/v3_prototype_d/fold0_metrics.json`
- `outputs/v3_prototype_d/fold1_metrics.json`
- `outputs/v3_prototype_d/fold2_metrics.json`
- `outputs/v3_prototype_d/fold3_metrics.json`
- `outputs/v3_prototype_d/fold4_metrics.json`
