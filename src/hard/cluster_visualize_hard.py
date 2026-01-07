import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.manifold import TSNE

def main():
    os.makedirs("results/plots_hard", exist_ok=True)

    # latent vectors
    Z = np.load("results/latent_hard.npy")

    # labels (encoded) + genre names
    y = np.load("data/hard/features/y_genre.npy")
    df_meta = pd.read_csv("data/hard/processed/songs_clean_balanced.csv")

    # latent size match করা (8000)
    n = Z.shape[0]
    df_meta = df_meta.iloc[:n].reset_index(drop=True)

    genres = df_meta["genre"].astype(str).str.lower().values

    # --- KMeans on latent ---
    k = len(pd.unique(genres))  # same as number of genres
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=50)
    clusters = kmeans.fit_predict(Z)

    np.save("results/cluster_labels.npy", clusters)

    # --- TSNE ---
    tsne = TSNE(n_components=2, random_state=42, perplexity=30, init="pca")
    Z2 = tsne.fit_transform(Z)

# prepare true genre ids for coloring
    genre_ids, genre_names = pd.factorize(genres)

# Plot 1: colored by cluster
    plt.figure()
    plt.scatter(Z2[:, 0], Z2[:, 1], c=clusters, s=6, cmap="tab10")
    plt.title("t-SNE of CVAE latent space (colored by cluster)")
    plt.savefig("results/plots_hard/tsne_by_cluster.png", dpi=200, bbox_inches="tight")
    plt.close()

    # Plot 2: colored by true genre
    plt.figure()
    plt.scatter(Z2[:, 0], Z2[:, 1], c=genre_ids, s=6, cmap="tab10")
    plt.title("t-SNE of CVAE latent (colored by true genre)")
    plt.savefig("results/plots_hard/tsne_by_true_genre.png", dpi=200, bbox_inches="tight")
    plt.close()

    # --- Cluster vs Genre distribution table ---
    df_out = pd.DataFrame({"genre": genres, "cluster": clusters})
    ctab = pd.crosstab(df_out["cluster"], df_out["genre"], normalize="index")

    # Heatmap-like plot using imshow (no seaborn)
    plt.figure(figsize=(10, 5))
    plt.imshow(ctab.values, aspect="auto")
    plt.yticks(range(ctab.shape[0]), ctab.index)
    plt.xticks(range(ctab.shape[1]), ctab.columns, rotation=45, ha="right")
    plt.title("Cluster distribution over genres (row-normalized)")
    plt.colorbar()
    plt.tight_layout()
    plt.savefig("results/plots_hard/cluster_genre_matrix.png", dpi=200)
    plt.close()

    ctab.to_csv("results/cluster_genre_matrix.csv")

    print("Saved: results/cluster_labels.npy")
    print("Saved plots in results/plots_hard/")
    print("Saved: results/cluster_genre_matrix.csv")

if __name__ == "__main__":
    main()