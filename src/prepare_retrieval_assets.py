

import os
import numpy as np
import torch
import faiss
import joblib
import pandas as pd
from PIL import Image
from tqdm import tqdm
from transformers import CLIPProcessor, CLIPModel
import umap

# ------------------------------------------------
# CONFIG
# ------------------------------------------------
IMAGE_DIR = "data/Train/images"
CSV_PATH = "data/annotations_train.csv"
OUTPUT_DIR = "src/embeddings2"

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ------------------------------------------------
# LOAD CSV (MATCHING YOUR COLUMN NAMES)
# ------------------------------------------------
df = pd.read_csv(CSV_PATH)

image_names = df["Image name"].values
grades = df["Retinopathy grade"].values

print("Total images:", len(image_names))

# ------------------------------------------------
# LOAD CLIP
# ------------------------------------------------
print("Loading CLIP model...")
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

model.to(DEVICE)
model.eval()

# ------------------------------------------------
# GENERATE IMAGE EMBEDDINGS
# ------------------------------------------------
print("Generating image embeddings...")

embeddings = []

for img_name in tqdm(image_names):
    img_path = os.path.join(IMAGE_DIR, img_name)

    if not os.path.exists(img_path):
        print(f"Missing image: {img_path}")
        continue

    image = Image.open(img_path).convert("RGB")

    inputs = processor(images=image, return_tensors="pt")
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

    with torch.no_grad():
        image_features = model.get_image_features(**inputs)

    # Normalize
    image_features = image_features / image_features.norm(dim=-1, keepdim=True)

    embeddings.append(image_features.cpu().numpy())

embeddings = np.vstack(embeddings).astype("float32")

print("Embedding shape:", embeddings.shape)

# ------------------------------------------------
# SAVE FILES
# ------------------------------------------------
np.save(os.path.join(OUTPUT_DIR, "image_embeddings2.npy"), embeddings)
np.save(os.path.join(OUTPUT_DIR, "grades2.npy"), grades)
np.save(os.path.join(OUTPUT_DIR, "image_names2.npy"), image_names)

print("Saved embeddings + metadata")

# ------------------------------------------------
# BUILD FAISS INDEX
# ------------------------------------------------
print("Building FAISS index...")

faiss.normalize_L2(embeddings)

index = faiss.IndexFlatIP(embeddings.shape[1])
index.add(embeddings)

faiss.write_index(index, os.path.join(OUTPUT_DIR, "image_only2.index"))

print("Saved image_only.index")

# ------------------------------------------------
# TRAIN UMAP
# ------------------------------------------------
print("Training UMAP...")

umap_model = umap.UMAP(
    n_components=3,
    n_neighbors=15,
    min_dist=0.1,
    metric="cosine",
    random_state=42
)

umap_model.fit(embeddings)

joblib.dump(umap_model, os.path.join(OUTPUT_DIR, "umap.pkl"))

print("Saved umap.pkl")

print("\n✅ Image-only retrieval assets ready.")
