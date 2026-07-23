# Odor family ontology (sensory-v3)

Status: **derived, validation in progress.** This document is the public,
versioned statement of the odor-family mapping used by the benchmark. The
machine-readable source of truth is `ODOR_FAMILIES` in
`src/dataset/sensory.py`; the ontology version is tied to the dataset
`schema_version` (`sensory-v3`). Any change to the mapping requires a new
ontology version and a fresh baseline before results are compared.

## Design intent

Raw descriptors from Leffingwell, GoodScents, and FlavorDB are normalised
(lowercase, whitespace collapsed, underscores to spaces) and mapped onto
twelve multi-label families. Terms that map to no family are retained as raw
evidence in `source_records.parquet` but produce no label. The twelve-family
granularity is deliberately finer than the earlier five-class prototype while
staying coarse enough for per-family support in the tens-to-hundreds.

Literature anchors used when drawing family boundaries:

- Dravnieks, *Atlas of Odor Character Profiles* (1985) — descriptor factor
  structure.
- Zarzo & Stanton, "Understanding the underlying dimensions in perfumers'
  odor perception space" (2009) — perfumery-note dimensionality.
- Chastrette, "Classification of odors and structure–odor relationships"
  (2002).
- Lee et al., "A principal odor map unifies diverse tasks in olfactory
  perception" (Science, 2023) — the 55-label GoodScents/Leffingwell label set
  this ontology coarsens.

## The twelve families

| Family | Members | Notes |
| --- | --- | --- |
| fruity | fruity, apple, apricot, banana, berry, cherry, citrus, coconut, grape, melon, peach, pear, pineapple, plum, strawberry, tropical | Citrus is folded in; a separate citrus family was considered but halves support. |
| floral | floral, geranium, hyacinth, jasmine, lavender, lily, muguet, neroli, orange flower, rose, violet | Follows perfumery white-floral / green-floral grouping. |
| green | cucumber, fresh, grassy, green, herbal, leafy, vegetable | "Fresh" is broad; retained here pending co-occurrence evidence. |
| woody | amber, balsamic, cedar, earthy, mossy, oak, pine, sandalwood, woody | Amber/balsamic are oriental-note territory in perfumery; grouped here for support. |
| fatty | buttery, cheesy, creamy, fatty, oily, rancid, waxy | Lipid/dairy notes. |
| sulfurous | alliaceous, brothy, burnt, cooked, garlic, meaty, onion, roasted, sulfur, sulfurous | Savory/thermal-process notes; strongest test family in v3. |
| spicy | anise, anisic, cinnamon, clove, ginger, nutmeg, pepper, spicy | Sweet-spice and pungent-spice merged. |
| sweet_aromatic | caramel, chocolate, cocoa, honey, sweet, vanilla | Gourmand notes. |
| nutty | almond, cocoa, nut skin, nutty, walnut | Tree-nut notes. |
| animalic | animal, fishy, leather, musky, sweaty | Musky is classically its own family; grouped here for support. |
| phenolic | medicinal, phenolic, smoky, tobacco | Smoke/creosote notes. |
| aldehydic | aldehydic, ethereal, metallic, pungent, sharp | Weakest coherence; see known issues. |

## Known issues (open defects, not hidden)

1. **`cocoa` belongs to two families** (`sweet_aromatic` and `nutty`). Every
   cocoa-carrying molecule is labelled with both families, artificially
   coupling them and inflating their co-occurrence. Decision for the next
   ontology revision: keep `cocoa` in `sweet_aromatic` (with chocolate) and
   leave `nutty` as tree-nut only. Impact quantification:
   `scripts/audit_odor_ontology.py` reports the affected molecule count.
2. **`aldehydic` is a grab-bag**: aldehydic/ethereal are citrus-adjacent
   aldehyde notes, while metallic and pungent are different perceptual
   qualities (trigeminal/mineral). Candidates for remapping to raw evidence
   or a separate family in the next revision.
3. **`musky` inside `animalic`** conflates the classic musk family with
   animal notes; a standalone musk family was rejected for support reasons.
4. **Coverage gaps**: minty/camphoraceous and waxy-green notes are currently
   unmapped by design; the audit script lists the top unmapped terms so the
   next revision can be evidence-driven.

## Validation plan

| Check | Tool | Question |
| --- | --- | --- |
| Shuffle negative control | `scripts/fingerprint_baseline.py --shuffle-ontology` | Do scores collapse when family grouping is destroyed? (They must.) |
| Co-occurrence lift | `scripts/audit_odor_ontology.py` | Do same-family descriptors co-occur on molecules more than cross-family ones? |
| Robustness to regrouping | rerun baseline with the revised ontology | Are headline conclusions (3D-vs-2D tie, taste gap) stable? |
| Confusion structure | per-label analysis of archived runs | Do confused families correspond to defensible neighbours? |

## Change policy

1. Any membership change bumps the ontology revision and is recorded here
   with rationale and evidence (audit output).
2. Downstream artefacts (prepared dataset, fold metrics, reports) name the
   ontology/schema version they were produced with.
3. Cross-version metric comparisons are marked non-comparable, as with the
   v2 → v3 transition.
