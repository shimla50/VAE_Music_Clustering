import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score, calinski_harabasz_score

DATA_DIR = "data/processed/easy"
RES_DIR = "results/easy"
PLOT_DIR = os.path.join(RES_DIR, "plots")
os.makedirs(PLOT_DIR, exist_ok=True)

LATENT_PATH = os.path.join(RES_DIR, "latent_easy.npy")
X_PATH = os.path.join(DATA_DIR, "X_easy.npy")
Y_PATH = os.path.join(DATA_DIR, "y_easy.npy")


def run_kmeans_metrics(X, k, seed=42):
    km = KMeans(n_clusters=k, random_state=seed, n_init="auto")
    pred = km.fit_predict(X)

    sil = float(silhouette_score(X, pred))
    ch = float(calinski_harabasz_score(X, pred))
    return pred, sil, ch


def plot_tsne(X, labels, title, out_path):
    tsne = TSNE(
        n_components=2,
        perplexity=30,
        init="pca",
        learning_rate="auto",
        random_state=42
    )
    Z2 = tsne.fit_transform(X)

    plt.figure()
    plt.scatter(Z2[:, 0], Z2[:, 1], c=labels, s=6, alpha=0.8, cmap="tab10")
    plt.title(title)
    plt.xlabel("t-SNE 1")
    plt.ylabel("t-SNE 2")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def main():
    # Load
    Z_vae = np.load(LATENT_PATH)        # (1577, 8)
    X = np.load(X_PATH)                # (1577, 3000)
    y_lang = np.load(Y_PATH)           # (1577,)
    print("Loaded Z_vae:", Z_vae.shape, "| X:", X.shape, "| y:", y_lang.shape)

    # k=2 (English vs Bangla)
    k = 2
    rows = []

    # ========== Method A: VAE latent + KMeans ==========
    pred_vae, sil_vae, ch_vae = run_kmeans_metrics(Z_vae, k)
    rows.append({
        "method": "VAE(latent)+KMeans",
        "k": k,
        "silhouette": sil_vae,
        "calinski_harabasz": ch_vae
    })
    print("VAE+KMeans -> silhouette:", sil_vae, "CH:", ch_vae)

    # Plot t-SNE latent space colored by clusters and true language
    plot_tsne(Z_vae, pred_vae, "Easy: t-SNE (VAE latent) colored by KMeans cluster",
              os.path.join(PLOT_DIR, "tsne_vae_by_cluster.png"))
    plot_tsne(Z_vae, y_lang, "Easy: t-SNE (VAE latent) colored by true language",
              os.path.join(PLOT_DIR, "tsne_vae_by_language.png"))

    # ========== Method B: PCA + KMeans baseline ==========
    pca = PCA(n_components=8, random_state=42)
    X_pca = pca.fit_transform(X)
    pred_pca, sil_pca, ch_pca = run_kmeans_metrics(X_pca, k)
    rows.append({
        "method": "PCA(8)+KMeans",
        "k": k,
        "silhouette": sil_pca,
        "calinski_harabasz": ch_pca
    })
    print("PCA+KMeans -> silhouette:", sil_pca, "CH:", ch_pca)

    plot_tsne(X_pca, pred_pca, "Easy: t-SNE (PCA features) colored by KMeans cluster",
              os.path.join(PLOT_DIR, "tsne_pca_by_cluster.png"))
    plot_tsne(X_pca, y_lang, "Easy: t-SNE (PCA features) colored by true language",
              os.path.join(PLOT_DIR, "tsne_pca_by_language.png"))

    # Save metrics
    out_csv = os.path.join(RES_DIR, "easy_metrics.csv")
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    print("✅ Saved metrics:", out_csv)
    print("✅ Saved plots in:", PLOT_DIR)


if __name__ == "__main__":
    main()