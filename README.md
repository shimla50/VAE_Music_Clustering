# 🎵 VAE Music Clustering

This project explores music clustering using Variational Autoencoders (VAE) and Conditional Variational Autoencoders (CVAE).  
The work is structured into three progressive tasks — **Easy**, **Medium**, and **Hard** — following increasing model complexity and dataset richness.

The goal is to learn meaningful latent representations of music using audio and lyrical information, and evaluate clustering quality using standard unsupervised metrics.

---

## 📌 Tasks Overview

### 🔹 Easy Task
- Basic feature extraction
- Traditional clustering techniques
- Introduction to unsupervised learning on music data

### 🔹 Medium Task
- Variational Autoencoder (VAE) for latent representation learning
- Audio + lyrics embeddings
- Clustering in learned latent space
- Quantitative evaluation using clustering metrics

### 🔹 Medium Task (Notebook workflow)

The medium task is implemented using Jupyter notebooks.

Run Jupyter Notebook:

```bash
python -m notebook

### 🔹 Hard Task
- **Conditional Variational Autoencoder (CVAE)**
- Multi-modal learning combining:
  - Audio features
  - Lyrics embeddings
  - Genre conditioning
- Extensive baseline comparisons:
  - PCA + KMeans
  - Autoencoder + KMeans
  - Direct feature clustering
- Detailed visualizations and reconstructions

---

## 📂 Dataset Sources

This project uses publicly available and widely used datasets:

- **Jamendo Lyrics Dataset**  
  https://github.com/f90/jamendolyrics

- **Jamendo Audio Dataset (Kaggle)**  
  https://www.kaggle.com/datasets/andradaolteanu/jamendo-music-dataset

> Raw audio files and large feature arrays are **not uploaded to GitHub** and are ignored via `.gitignore`.

---

## 🗂️ Project Structure

VAE_Music_Clustering/
│
├── data/
│ ├── raw/
│ │ ├── easy/
│ │ ├── medium/
│ │ └── hard/
│ └── processed/
│ ├── easy/
│ ├── medium/
│ └── hard/
│
├── notebooks/
│ ├── easy/
│ ├── medium/
│ └── hard/
│
├── src/
│ ├── common/ # Shared utilities (dataset, VAE, clustering, evaluation)
│ ├── easy/
│ ├── medium/
│ └── hard/
│
├── results/
│ ├── easy/
│ ├── medium/
│ │ ├── clustering_metrics.csv
│ │ └── plots/
│ └── hard/
│ └── plots/
│
├── requirements.txt
├── .gitignore
└── README.md


---

## ⚙️ Installation & Setup

### 1️⃣ Create virtual environment (recommended)
```bash
python -m venv venv
source venv/bin/activate   # Linux / Mac
venv\Scripts\activate      # Windows

