# Cross-sensory fold summary

Folds: 0, 1, 2, 3, 4 (n=5)

## Protocol

- Core taste labels: sweet, bitter, umami
- Low-shot labels: sour, salty
- Alignment: `{"C": 1.0, "class_weight": "balanced", "model": "morgan_fingerprint_logreg", "n_bits": 2048, "radius": 2, "solver": "liblinear"}`

## Held-out test metrics

| Task | Metric | Mean ± SD | Per fold |
| --- | --- | ---: | --- |
| odor | aldehydic | 0.4090 ± 0.0534 | 0.3222, 0.4310, 0.3989, 0.4309, 0.4619 |
| odor | animalic | 0.2598 ± 0.0705 | 0.1923, 0.3421, 0.3288, 0.2059, 0.2301 |
| odor | fatty | 0.6011 ± 0.0236 | 0.5677, 0.6093, 0.5863, 0.6174, 0.6246 |
| odor | floral | 0.5443 ± 0.0627 | 0.6468, 0.5459, 0.5390, 0.5075, 0.4823 |
| odor | fruity | 0.7158 ± 0.0462 | 0.6549, 0.7661, 0.7175, 0.6866, 0.7538 |
| odor | green | 0.6408 ± 0.0426 | 0.5667, 0.6686, 0.6472, 0.6697, 0.6518 |
| odor | macro | 0.4849 ± 0.0069 | 0.4831, 0.4968, 0.4827, 0.4827, 0.4790 |
| odor | nutty | 0.3478 ± 0.1045 | 0.1946, 0.3687, 0.2974, 0.4231, 0.4554 |
| odor | phenolic | 0.2257 ± 0.1115 | 0.4175, 0.1852, 0.2105, 0.1274, 0.1878 |
| odor | spicy | 0.3022 ± 0.0819 | 0.4334, 0.2353, 0.2977, 0.3131, 0.2314 |
| odor | sulfurous | 0.7297 ± 0.0269 | 0.7599, 0.7465, 0.7368, 0.7125, 0.6930 |
| odor | sweet_aromatic | 0.5258 ± 0.0510 | 0.6052, 0.5277, 0.5062, 0.5249, 0.4649 |
| odor | woody | 0.5164 ± 0.0505 | 0.4364, 0.5350, 0.5261, 0.5739, 0.5108 |
| taste | bitter | 0.8351 ± 0.0489 | 0.7715, 0.8448, 0.8053, 0.8544, 0.8995 |
| taste | macro | 0.7855 ± 0.0365 | 0.7341, 0.7910, 0.8034, 0.7685, 0.8307 |
| taste | sweet | 0.8358 ± 0.0448 | 0.7717, 0.8457, 0.8481, 0.8194, 0.8942 |
| taste | umami | 0.6856 ± 0.0471 | 0.6591, 0.6824, 0.7568, 0.6316, 0.6984 |
| combined | score | 0.6352 ± 0.0182 | 0.6086, 0.6439, 0.6431, 0.6256, 0.6549 |

## Inputs

- `outputs/v3_fingerprint_lr/fold0_metrics.json`
- `outputs/v3_fingerprint_lr/fold1_metrics.json`
- `outputs/v3_fingerprint_lr/fold2_metrics.json`
- `outputs/v3_fingerprint_lr/fold3_metrics.json`
- `outputs/v3_fingerprint_lr/fold4_metrics.json`
