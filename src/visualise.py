import numpy as np
import umap
import plotly.express as px
import os

from evaluate import precision_at_k,mean_average_precision
def create_3d_umap(embedding_path="embeddings"):

    image_embeddings = np.load(os.path.join(embedding_path, "image_embeddings.npy"))
    text_embeddings = np.load(os.path.join(embedding_path, "text_embeddings.npy"))
    grades = np.load(os.path.join(embedding_path, "grades.npy"))

    # Weighted fusion
    combined = 0.7 * image_embeddings + 0.3 * text_embeddings

    # Normalize again (important)
    combined = combined / np.linalg.norm(combined, axis=1, keepdims=True)

    reducer = umap.UMAP(
        n_components=3,
        n_neighbors=20,
        min_dist=0.2,
        metric="cosine",
        random_state=42
    )

    embeddings_3d = reducer.fit_transform(combined)

    fig = px.scatter_3d(
        x=embeddings_3d[:, 0],
        y=embeddings_3d[:, 1],
        z=embeddings_3d[:, 2],
        color=grades.astype(str),
        title="3D Multimodal Embedding Space (Image + Text)"
    )
    p5 = precision_at_k(combined, grades, k=5)
    map_score = mean_average_precision(combined, grades)

    print("Precision@5:", p5)
    print("mAP:", map_score)
    fig.update_traces(marker=dict(size=4))
    fig.show()


if __name__ == "__main__":
    create_3d_umap()

    
