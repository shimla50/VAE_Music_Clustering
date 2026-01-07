# 🎵 VAE Music Clustering (Easy • Medium • Hard)

This repository contains an academic project on **music clustering using Variational Autoencoders (VAE)**.  
The work is organized into three levels: **Easy**, **Medium**, and **Hard**, showing progressive improvements from basic clustering to **multi-modal CVAE-based clustering**.

---

## ✅ Tasks Overview

Easy Task Results

Dataset: English + Bangla lyrics subset (n=1577)

Features: TF-IDF (3000 dims)

VAE latent size: 8

Clustering: KMeans (k=2)

Metrics:

VAE(latent)+KMeans: Silhouette = 0.144, Calinski-Harabasz = 272.07

PCA(8)+KMeans: Silhouette = 0.424, Calinski-Harabasz = 1079.14

Visualizations: t-SNE plots saved in results/easy/plots/

Short interpretation (1–2 lines)
PCA baseline performs better because TF-IDF space already separates languages strongly, while the VAE is trained only for reconstruction and may not preserve language-separating structure in latent space.

### Medium Task
- Notebook-based workflow: data prep → lyrics embedding → audio MFCC → VAE latent clustering
- Notebooks: `notebooks/medium/`
- Outputs: `results/medium/`

### Hard Task
- **Conditional VAE (CVAE)** + multi-modal clustering using **audio + lyrics + genre**
- Baselines: **PCA + KMeans**, **Autoencoder + KMeans**, **Direct feature clustering**
- Visualizations: t-SNE/UMAP, cluster-genre distribution, reconstruction error analysis
- Scripts: `src/hard/`
- Outputs: `results/hard/`

---

## 📂 Dataset Sources

- **Jamendo Lyrics Dataset (lyrics + language)**  
  https://github.com/f90/jamendolyrics

- **Jamendo Audio + Genre (Kaggle)**  
  https://www.kaggle.com/datasets/andradaolteanu/jamendo-music-dataset

> Raw audio and large feature/model files are not stored in GitHub (ignored via `.gitignore`).

---

## 🗂️ Repository Structure

VAE_Music_Clustering/
├── data/
│ ├── raw/
│ └── processed/
├── notebooks/
│ ├── easy/
│ ├── medium/
│ └── hard/
├── src/
│ ├── common/
│ ├── easy/
│ ├── medium/
│ └── hard/
├── results/
│ ├── easy/
│ ├── medium/
│ └── hard/
├── requirements.txt
├── .gitignore
└── README.md


---

## ⚙️ Installation & Setup

Install dependencies:

```bash
pip install -r requirements.txt


