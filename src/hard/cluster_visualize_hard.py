import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.manifold import TSNE

RESULTS_DIR = os.path.join("results", "hard")
PLOTS_DIR = os.path.join(RESULTS_DIR, "plots")
os.makedirs(PLOTS_DIR, exist_ok=True)

def main():
    Z = np.load(os.path.join(RESULTS_DIR, "latent_hard.npy"))

    df_meta = pd.read_csv(os.path.join("data", "hard", "processed", "clean_balanced_songs.csv"))

    n = Z.shape[0]
    df_meta = df_meta.iloc[:n].reset_index(drop=True)

    genres = df_meta["genre"].astype(str).str.lower().values

    k = len(pd.unique(genres))
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=50)
    clusters = kmeans.fit_predict(Z)

    np.save(os.path.join(RESULTS_DIR, "cluster_labels.npy"), clusters)

    tsne = TSNE(n_components=2, random_state=42, perplexity=30, init="pca")
    Z2 = tsne.fit_transform(Z)

    genre_ids, _ = pd.factorize(genres)

    plt.figure()
    plt.scatter(Z2[:, 0], Z2[:, 1], c=clusters, s=6, cmap="tab10")
    plt.title("t-SNE of CVAE latent space (colored by cluster)")
    p1 = os.path.join(PLOTS_DIR, "tsne_by_cluster.png")
    plt.savefig(p1, dpi=200, bbox_inches="tight")
    plt.close()

    plt.figure()
    plt.scatter(Z2[:, 0], Z2[:, 1], c=genre_ids, s=6, cmap="tab10")
    plt.title("t-SNE of CVAE latent (colored by true genre)")
    p2 = os.path.join(PLOTS_DIR, "tsne_by_true_genre.png")
    plt.savefig(p2, dpi=200, bbox_inches="tight")
    plt.close()

    df_out = pd.DataFrame({"genre": genres, "cluster": clusters})
    ctab = pd.crosstab(df_out["cluster"], df_out["genre"], normalize="index")

    plt.figure(figsize=(10, 5))
    plt.imshow(ctab.values, aspect="auto")
    plt.yticks(range(ctab.shape[0]), ctab.index)
    plt.xticks(range(ctab.shape[1]), ctab.columns, rotation=45, ha="right")
    plt.title("Cluster distribution over genres (row-normalized)")
    plt.colorbar()
    plt.tight_layout()
    p3 = os.path.join(PLOTS_DIR, "cluster_genre_matrix.png")
    plt.savefig(p3, dpi=200)
    plt.close()

    ctab.to_csv(os.path.join(RESULTS_DIR, "cluster_genre_matrix.csv"))

    print("Saved:", os.path.join(RESULTS_DIR, "cluster_labels.npy"))
    print("Saved plots in:", PLOTS_DIR)
    print("Saved:", os.path.join(RESULTS_DIR, "cluster_genre_matrix.csv"))

if __name__ == "__main__":
    main()
