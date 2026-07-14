"""5-class odor label mapping for Leffingwell dataset."""

CATEGORY_MAP_V2 = {
    "fruity": [
        "fruity", "apple", "tropical", "berry", "citrus", "peach", "pineapple",
        "banana", "coconut", "grape", "melon", "pear", "plum", "strawberry",
        "cherry", "apricot", "pine",
        "green", "grassy", "herbal", "leafy", "cucumber", "vegetable", "fresh",
        "winey", "fermented",
    ],
    "sweet": [
        "sweet", "creamy", "vanilla", "caramel", "honey", "buttery", "milky",
        "jammy", "marshmallow", "warm", "chocolate", "cocoa",
        "floral", "rose", "jasmine", "lavender", "violet", "lily", "hyacinth",
        "muguet", "geranium", "neroli", "orange",
    ],
    "sulfurous": [
        "sulfurous", "alliaceous", "roasted", "onion", "garlic", "meaty",
        "savory", "cooked", "brothy", "coffee", "burnt", "pungent",
        "spicy", "clove", "cinnamon", "pepper", "ginger", "nutmeg",
        "anise", "anisic",
    ],
    "woody": [
        "woody", "balsamic", "cedar", "sandalwood", "oak", "amber",
        "pine", "camphoreous", "dry",
        "earthy", "musty", "mushroom", "dusty", "soil", "mossy", "phenolic",
        "medicinal", "smoky", "leather", "tobacco", "tea", "hay",
        "animal", "fishy", "gasoline", "sour",
    ],
    "fatty": [
        "fatty", "oily", "waxy", "cheesy", "rancid", "butter",
    ],
}

ALL_5_CLASSES = list(CATEGORY_MAP_V2.keys())

import numpy as np


def get_category_set(labels):
    cats = set()
    for cat, keywords in CATEGORY_MAP_V2.items():
        for kw in keywords:
            if kw in labels:
                cats.add(cat)
                break
    return cats


def labels_to_vector(labels):
    cat_set = get_category_set(labels)
    return np.array([1 if c in cat_set else 0 for c in ALL_5_CLASSES], dtype=np.float32)


def encode_labels(df):
    df = df.copy()
    df["y"] = df["labels"].apply(labels_to_vector)
    return df


def label_distribution(df):
    y_stack = np.stack(df["y"].values)
    print(f"{'Class':<12} {'Count':>6}")
    print("-" * 20)
    for i, cls in enumerate(ALL_5_CLASSES):
        print(f"{cls:<12} {int(y_stack[:, i].sum()):>6}")
    print("-" * 20)
    multi = (y_stack.sum(axis=1) > 1).sum()
    print(f"Multi-label:  {multi}")
    print(f"Total:        {len(df)}")
