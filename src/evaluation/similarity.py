import pickle
from pathlib import Path
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

EMBEDDING_DIR = Path(__file__).resolve().parents[2] / "data" / "processed" / "embeddings"


def load_embeddings(path=None):
    if path is None:
        path = EMBEDDING_DIR / "leffingwell_embeddings.pkl"
    with open(path, "rb") as f:
        return pickle.load(f)


def top_similar(embeddings, query, top_k=10):
    compounds = list(embeddings.keys())
    vecs = np.stack(list(embeddings.values()))
    sim_matrix = cosine_similarity(vecs)

    idx = compounds.index(query)
    sims = sim_matrix[idx]
    ranked = np.argsort(sims)[::-1]

    results = []
    for i in ranked[: top_k + 1]:
        if compounds[i] != query:
            results.append((compounds[i], sims[i]))
    return results


def query_similar(embeddings, query, top_k=10):
    results = top_similar(embeddings, query, top_k)
    print(f"\nTop-{top_k} similar to '{query}':")
    for name, sim in results:
        print(f"  {sim:.4f}  {name}")
    return results
