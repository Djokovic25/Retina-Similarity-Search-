

import os
import torch
import numpy as np
import streamlit as st
import faiss
import joblib
import plotly.express as px
import plotly.graph_objects as go
from PIL import Image
from transformers import CLIPProcessor, CLIPModel

# ------------------------------------------------
# CONFIG
# ------------------------------------------------
DATA_DIR = "src/embeddings2"
IMAGE_DIR = "data/Train/images"

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"

st.set_page_config(layout="wide")
st.title("🧠 Diabetic Retinopathy Image Retrieval System")

# ------------------------------------------------
# LOAD ASSETS
# ------------------------------------------------
@st.cache_resource
def load_assets():
    embeddings = np.load(os.path.join(DATA_DIR, "image_embeddings2.npy"))
    grades = np.load(os.path.join(DATA_DIR, "grades2.npy"))
    image_names = np.load(
        os.path.join(DATA_DIR, "image_names2.npy"),
        allow_pickle=True
    )

    index = faiss.read_index(os.path.join(DATA_DIR, "image_only2.index"))
    umap_model = joblib.load(os.path.join(DATA_DIR, "umap.pkl"))

    return embeddings, grades, image_names, index, umap_model


embeddings, grades, image_names, index, umap_model = load_assets()

# ------------------------------------------------
# LOAD CLIP IMAGE MODEL
# ------------------------------------------------
@st.cache_resource
def load_model():
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    model.to(DEVICE)
    model.eval()

    return model, processor


clip_model, clip_processor = load_model()

# ------------------------------------------------
# SIDEBAR
# ------------------------------------------------
st.sidebar.header("Settings")
top_k = st.sidebar.slider("Top-K Retrieval", 3, 10, 5)

# ------------------------------------------------
# UPLOAD IMAGE
# ------------------------------------------------
uploaded_file = st.file_uploader(
    "Upload Fundus Image",
    type=["jpg", "png", "jpeg"]
)

if uploaded_file is not None:

    col1, col2 = st.columns(2)

    image = Image.open(uploaded_file).convert("RGB")
    col1.image(image, caption="Query Image", use_column_width=True)

    # ------------------------------------------------
    # CREATE IMAGE EMBEDDING
    # ------------------------------------------------
    inputs = clip_processor(images=image, return_tensors="pt")
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

    with torch.no_grad():
        image_features = clip_model.get_image_features(**inputs)

    # Normalize
    image_features = image_features / image_features.norm(dim=-1, keepdim=True)
    query_embedding = image_features.cpu().numpy().astype("float32")

    st.write("Query embedding norm:", np.linalg.norm(query_embedding))

    # ------------------------------------------------
    # RETRIEVAL
    # ------------------------------------------------
    D, I = index.search(query_embedding, top_k)

    st.subheader("🔎 Top Similar Retrieved Images")

    cols = st.columns(top_k)

    for i in range(top_k):
        idx = I[0][i]
        img_path = os.path.join(IMAGE_DIR, image_names[idx])
        retrieved_img = Image.open(img_path)

        cols[i].image(
            retrieved_img,
            caption=f"Grade: {grades[idx]} | Similarity: {D[0][i]:.3f}",
            use_column_width=True
        )
    # ------------------------------------------------
    # KNN PREDICTION
    # ------------------------------------------------
    st.subheader("📊 KNN-Based Prediction")

    neighbor_indices = I[0]
    neighbor_grades = grades[neighbor_indices]
    neighbor_sims = D[0]

    # Majority vote
    values, counts = np.unique(neighbor_grades, return_counts=True)
    majority_grade = values[np.argmax(counts)]

    # Weighted soft grade
    weighted_grade = np.sum(neighbor_sims * neighbor_grades) / np.sum(neighbor_sims)

    colA, colB = st.columns(2)

    colA.metric("Majority Grade (KNN)", int(majority_grade))
    colB.metric("Weighted Soft Grade", f"{weighted_grade:.2f}")

    # Grade distribution bar chart
    import pandas as pd
    dist_df = pd.DataFrame({
        "Grade": neighbor_grades.astype(str)
    })

    st.bar_chart(dist_df["Grade"].value_counts().sort_index())



    # ------------------------------------------------
    # SIMILARITY TO GRADE CENTROIDS
    # ------------------------------------------------
    st.subheader("🧭 Similarity to Grade Centroids")

    unique_grades = np.unique(grades)
    centroid_sims = []

    for g in unique_grades:
        grade_embeddings = embeddings[grades == g]
        centroid = grade_embeddings.mean(axis=0)

        # Normalize centroid
        centroid = centroid / np.linalg.norm(centroid)

        sim = np.dot(query_embedding[0], centroid)
        centroid_sims.append(sim)

    centroid_df = pd.DataFrame({
        "Grade": unique_grades.astype(str),
        "Similarity": centroid_sims
    })
    

    st.line_chart(centroid_df.set_index("Grade"))
    # ------------------------------------------------
    # DISEASE SEVERITY AXIS
    # ------------------------------------------------
    st.subheader("📈 Disease Severity Axis Projection")

    # Compute centroids for Grade 0 and Grade 4
    grade0_centroid = embeddings[grades == 0].mean(axis=0)
    grade4_centroid = embeddings[grades == 4].mean(axis=0)

    # Normalize
    grade0_centroid /= np.linalg.norm(grade0_centroid)
    grade4_centroid /= np.linalg.norm(grade4_centroid)

    severity_direction = grade4_centroid - grade0_centroid
    severity_direction /= np.linalg.norm(severity_direction)

    # Project query onto severity axis
    severity_score = np.dot(query_embedding[0], severity_direction)

    st.metric("Severity Axis Score", f"{severity_score:.3f}")

    st.caption(
        "Projection of query embedding onto the Grade 0 → Grade 4 progression direction."
    )




    # ------------------------------------------------
    # 3D UMAP VISUALIZATION
    # ------------------------------------------------
    st.subheader("🌌 3D Image Embedding Space")

    reduced = umap_model.transform(embeddings)
    query_3d = umap_model.transform(query_embedding)

    import pandas as pd

    df = pd.DataFrame({
        "x": reduced[:, 0],
        "y": reduced[:, 1],
        "z": reduced[:, 2],
        "grade": grades.astype(str)
    })

    fig = px.scatter_3d(
        df,
        x="x",
        y="y",
        z="z",
        color="grade",
        opacity=0.6
    )

    fig.add_trace(
        go.Scatter3d(
            x=[query_3d[0][0]],
            y=[query_3d[0][1]],
            z=[query_3d[0][2]],
            mode="markers",
            marker=dict(size=10, color="#D8FF00"),
            name="Query"
        )
    )

    fig.update_layout(height=700)

    st.plotly_chart(fig, use_container_width=True)

else:
    st.info("Upload a retinal fundus image to begin retrieval.")

