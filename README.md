# 🧠 Retina Similarity Search 
**Embedding-Based Diabetic Retinopathy Retrieval & Severity Modeling** 

--- 

## 📌 Overview 

An interactive application for **diabetic retinopathy (DR) grading** using **embedding-space retrieval** instead of traditional supervised classification.

- Retinal fundus images are encoded into **512-dimensional CLIP embeddings** and indexed using **FAISS** for fast similarity search.
- Severity is inferred using **K-Nearest Neighbors (KNN)** and **geometric projections** in latent space.
- Disease structure is visualized using **3D UMAP**. This reframes diagnosis as **geometry in representation space** rather than rigid classification.

--- 

![Demo Screenshot2](https://github.com/Djokovic25/Retina-Similarity-Search-/blob/9beb6504c817117bb779d3e51b775823e861493b/assets/Screenshot%202026-02-16%20at%203.28.56%E2%80%AFPM.png)

**Why?**
- Disease progression is continuous, not discrete  
- Similar past cases provide interpretability  
- Strong representations can replace heavy supervised training  
- Embedding geometry reveals structural patterns in severity

---
![KNN](https://github.com/Djokovic25/Retina-Similarity-Search-/blob/9beb6504c817117bb779d3e51b775823e861493b/assets/Screenshot%202026-02-16%20at%203.29.07%E2%80%AFPM.png)

## ✨ Core Components

### 🔹 Image Embeddings
- CLIP (ViT-B/32) for **512D image representations**  
- L2-normalized for cosine similarity  

### 🔹 FAISS Retrieval
- Fast nearest neighbor search  
- Top-K similar retinal images returned  
- Cosine similarity scores displayed  

### 🔹 KNN-Based Severity Inference
- Majority vote grade  
- Weighted soft severity score  
- Grade distribution visualization  

### 🔹 Centroid Similarity Analysis
- Computes similarity of query image to each grade centroid  
- Reveals geometric relationships across severity levels  

### 🔹 Disease Severity Axis
Defines a progression direction in embedding space:

The query image is projected onto this axis to obtain a **continuous severity score**.

### 🔹 3D UMAP Visualization
- Entire dataset projected into 3D  
- Color-coded by DR grade  
- Query image highlighted  
- Enables visual exploration of disease structure  

---

## 💻 Technologies Used
- **PyTorch**  
- **HuggingFace Transformers (CLIP)**  
- **FAISS**  
- **UMAP**  
- **Streamlit**  
- **Plotly**  

---

## 📂 Project Structure
```text
Retina-Similarity-Search/
│
├── app.py
├── requirements.txt
├── README.md
│
├── data/
│   └── Train/images/
│
└── src/
    └── embeddings2/
        ├── image_embeddings2.npy
        ├── grades2.npy
        ├── image_names2.npy
        ├── image_only2.index
        └── umap.pkl
```

![Embedding Visualisation](https://github.com/Djokovic25/Retina-Similarity-Search-/blob/9beb6504c817117bb779d3e51b775823e861493b/assets/Screenshot%202026-02-16%20at%203.31.11%E2%80%AFPM.png)

## 🚀 Installation & Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/your-username/Retina-Similarity-Search.git
   cd Retina-Similarity-Search
   python -m venv venv
   source venv/bin/activate   # On Windows: venv\Scripts\activate
   pip install -r requirements.txt

2. **How to Use**
   ```bash
    streamlit run app.py

   
   


