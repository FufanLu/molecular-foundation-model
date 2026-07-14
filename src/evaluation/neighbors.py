import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from .similarity import load_embeddings


def all_neighbors(embeddings, top_k=10):
    compounds = list(embeddings.keys())
    vecs = np.stack(list(embeddings.values()))
    sim_matrix = cosine_similarity(vecs)

    results = {}
    for i, compound in enumerate(compounds):
        sims = sim_matrix[i]
        ranked = np.argsort(sims)[::-1][1 : top_k + 1]
        neighbors = [(compounds[j], float(sims[j])) for j in ranked]
        results[compound] = neighbors

    return results


def neighbor_report(embeddings, df, queries, top_k=10):
    compounds = list(embeddings.keys())
    vecs = np.stack(list(embeddings.values()))
    sim_matrix = cosine_similarity(vecs)

    lookup = dict(zip(df["compound"], df["labels"]))

    for query in queries:
        idx = compounds.index(query)
        sims = sim_matrix[idx]
        ranked = np.argsort(sims)[::-1][1 : top_k + 1]

        print(f"\n{'='*60}")
        query_labels = lookup.get(query, [])
        print(f"Query: {query}")
        if query_labels:
            print(f"Labels: {', '.join(query_labels)}")
        print(f"{'─'*60}")
        print(f"{'Rank':<6}{'Neighbor':<35}{'Sim':>8}  Labels")
        print(f"{'─'*60}")

        for r, j in enumerate(ranked, 1):
            neighbor = compounds[j]
            sim = float(sims[j])
            labels = lookup.get(neighbor, [])
            label_str = ", ".join(labels)
            print(f"{r:<6}{neighbor:<35}{sim:>8.4f}  {label_str}")
