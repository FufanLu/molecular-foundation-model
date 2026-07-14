import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


def jaccard(set1, set2):
    if not set1 and not set2:
        return 1.0
    inter = len(set1 & set2)
    union = len(set1 | set2)
    return inter / union if union > 0 else 0.0


def label_consistency(embeddings, df, queries, top_k=10):
    compounds = list(embeddings.keys())
    vecs = np.stack(list(embeddings.values()))
    sim_matrix = cosine_similarity(vecs)
    lookup = dict(zip(df["compound"], df["labels"]))

    for query in queries:
        idx = compounds.index(query)
        sims = sim_matrix[idx]
        ranked = np.argsort(sims)[::-1][1 : top_k + 1]

        query_labels = set(lookup.get(query, []))

        print(f"\n{'='*60}")
        print(f"Query: {query}")
        print(f"Own labels: {sorted(query_labels)}")
        print(f"{'─'*60}")
        print(f"{'Neighbor':<35}{'Neighbor labels':<45}{'Jaccard':>8}")
        print(f"{'─'*60}")

        total_jaccard = 0.0
        for j in ranked:
            neighbor = compounds[j]
            neighbor_labels = set(lookup.get(neighbor, []))
            jac = jaccard(query_labels, neighbor_labels)
            total_jaccard += jac
            label_str = ", ".join(sorted(neighbor_labels))
            print(f"{neighbor:<35}{label_str:<45}{jac:>8.4f}")

        avg = total_jaccard / top_k
        print(f"{'─'*60}")
        print(f"Avg Jaccard: {avg:.4f}")


def global_consistency(embeddings, df, top_k=10, sample_size=None):
    compounds_list = list(embeddings.keys())
    if sample_size and sample_size < len(compounds_list):
        rng = np.random.RandomState(42)
        indices = rng.choice(len(compounds_list), sample_size, replace=False)
        compounds_list = [compounds_list[i] for i in indices]

    vecs = np.stack(list(embeddings.values()))
    sim_matrix = cosine_similarity(vecs)
    lookup = dict(zip(df["compound"], df["labels"]))
    all_labels = sorted(set(l for lbls in lookup.values() for l in lbls))

    scores = []
    for compound in compounds_list:
        idx = list(embeddings.keys()).index(compound)
        sims = sim_matrix[idx]
        ranked = np.argsort(sims)[::-1][1 : top_k + 1]

        query_labels = set(lookup.get(compound, []))
        if not query_labels:
            continue

        for j in ranked:
            neighbor = list(embeddings.keys())[j]
            neighbor_labels = set(lookup.get(neighbor, []))
            jac = jaccard(query_labels, neighbor_labels)
            scores.append(jac)

    avg_score = np.mean(scores) if scores else 0.0
    print(f"\nGlobal Label Consistency (top-{top_k} neighbors):")
    print(f"  Sampled: {len(compounds_list)} molecules")
    print(f"  Avg Jaccard: {avg_score:.4f}")
    print(f"  All unique labels: {len(all_labels)}")
    return avg_score
