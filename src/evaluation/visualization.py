import numpy as np
import matplotlib.pyplot as plt
import umap
from pathlib import Path

OUTPUT_DIR = Path(__file__).resolve().parents[2] / "outputs"

CATEGORY_MAP = {
    "fruity": ["fruity", "apple", "tropical", "berry", "citrus", "peach", "pineapple",
               "banana", "coconut", "grape", "melon", "pear", "plum", "strawberry",
               "cherry", "apricot", "pine", "ripe", "anise"],
    "green": ["green", "grassy", "herbal", "leafy", "cucumber", "vegetable", "fresh"],
    "woody": ["woody", "balsamic", "cedar", "sandalwood", "oak", "amber"],
    "floral": ["floral", "rose", "jasmine", "lavender", "violet", "lily", "hyacinth",
               "muguet", "geranium", "neroli"],
    "spicy": ["spicy", "clove", "cinnamon", "pepper", "ginger", "nutmeg"],
    "sulfurous": ["sulfurous", "alliaceous", "roasted", "onion", "garlic", "meaty",
                  "savory", "cooked", "brothy", "coffee", "cocoa", "chocolate"],
    "sweet": ["sweet", "caramel", "honey", "vanilla", "buttery", "creamy", "milky",
              "jammy", "marshmallow"],
    "fatty": ["fatty", "oily", "waxy", "cheesy", "rancid", "butter"],
    "minty": ["minty", "cooling", "camphor", "menthol", "eucalyptus"],
    "earthy": ["earthy", "musty", "mushroom", "dusty", "soil", "mossy", "phenolic",
               "medicinal", "smoky", "leather", "tobacco", "tea"],
}


def _get_category(labels):
    if isinstance(labels, str):
        return "unknown"
    for cat, keywords in CATEGORY_MAP.items():
        for kw in keywords:
            if kw in labels:
                return cat
    return "unknown"


def plot_umap(embeddings, df, save_path=None, width=12, height=10):
    compounds = list(embeddings.keys())
    vecs = np.stack(list(embeddings.values()))
    lookup = dict(zip(df["compound"], df["labels"]))

    reducer = umap.UMAP(n_components=2, random_state=42, n_neighbors=15, min_dist=0.1)
    coords = reducer.fit_transform(vecs)

    categories = []
    for compound in compounds:
        labels = lookup.get(compound, [])
        cat = _get_category(labels)
        categories.append(cat)

    unique_cats = sorted(set(categories))
    colors = plt.cm.tab10(np.linspace(0, 1, len(unique_cats)))
    color_map = dict(zip(unique_cats, colors))

    fig, ax = plt.subplots(figsize=(width, height))

    for cat in unique_cats:
        if cat == "unknown":
            continue
        mask = [c == cat for c in categories]
        if sum(mask) == 0:
            continue
        ax.scatter(
            coords[mask, 0],
            coords[mask, 1],
            c=[color_map[cat]],
            label=f"{cat} ({sum(mask)})",
            alpha=0.6,
            s=10,
        )

    unknown_mask = [c == "unknown" for c in categories]
    if sum(unknown_mask) > 0:
        ax.scatter(
            coords[unknown_mask, 0],
            coords[unknown_mask, 1],
            c="lightgray",
            label=f"other ({sum(unknown_mask)})",
            alpha=0.3,
            s=5,
        )

    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1), fontsize=9, markerscale=2)
    ax.set_title("Uni-Mol Molecular Embedding Space (UMAP)", fontsize=14)
    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved to {save_path}")
    else:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        fp = OUTPUT_DIR / "umap_embedding.png"
        plt.savefig(fp, dpi=150, bbox_inches="tight")
        print(f"Saved to {fp}")

    plt.show()

    print(f"\nCategory distribution:")
    for cat in unique_cats:
        count = sum(1 for c in categories if c == cat)
        print(f"  {cat:<12} {count:>4}")
