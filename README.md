# 🎵 VAE Music Clustering (Easy • Medium • Hard)

This repository contains an academic project on **music clustering using Variational Autoencoders (VAE)**.  
The work is organized into three levels: **Easy**, **Medium**, and **Hard**, showing progressive improvements from basic clustering to **multi-modal CVAE-based clustering**.

---

## ✅ Tasks Overview

### Easy Task
- Basic preprocessing and clustering (traditional baseline level)
- Folder: `src/easy/` (if implemented) and results under `results/easy/`

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


