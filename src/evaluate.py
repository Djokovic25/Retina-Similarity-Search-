import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

def precision_at_k(embeddings, labels, k=5):
    sims = cosine_similarity(embeddings)
    np.fill_diagonal(sims, -1)

    precisions = []

    for i in range(len(embeddings)):
        top_k = sims[i].argsort()[-k:]
        correct = sum(labels[j] == labels[i] for j in top_k)
        precisions.append(correct / k)

    return np.mean(precisions)


def mean_average_precision(embeddings, labels):
    sims = cosine_similarity(embeddings)
    np.fill_diagonal(sims, -1)

    APs = []

    for i in range(len(embeddings)):
        sorted_idx = sims[i].argsort()[::-1]
        relevant = (labels == labels[i]).astype(int)

        hits = 0
        precisions = []

        for rank, idx in enumerate(sorted_idx):
            if relevant[idx] == 1:
                hits += 1
                precisions.append(hits / (rank + 1))

        if precisions:
            APs.append(np.mean(precisions))

    return np.mean(APs)
