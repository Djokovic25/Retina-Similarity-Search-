import numpy as np
import faiss
import os

def build_faiss_index(embedding_path="embeddings"):
    image_embeddings = np.load(os.path.join(embedding_path, "image_embeddings.npy"))

    dimension = image_embeddings.shape[1]

    index = faiss.IndexFlatIP(dimension)  # cosine similarity (since normalized)
    index.add(image_embeddings)

    faiss.write_index(index, os.path.join(embedding_path, "faiss_index.index"))

    print("FAISS index built and saved.")

def search(query_embedding, k=5, embedding_path="embeddings"):
    index = faiss.read_index(os.path.join(embedding_path, "faiss_index.index"))

    scores, indices = index.search(query_embedding, k)
    return scores, indices


if __name__ == "__main__":
    build_faiss_index()