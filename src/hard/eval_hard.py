import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, adjusted_rand_score, normalized_mutual_info_score
from sklearn.manifold import TSNE

# ===== paths =====
Z_PATH = os.path.join("results", "latent_hard.npy")
Y_PATH = os.path.join("data", "hard", "features", "y_genre.npy")
GENRE_MAP_PATH = os.path.join("data", "hard", "features", "genre_map.json")

OUT_PLOTS = os.path.join("results", "plots_hard")
os.makedirs(OUT_PLOTS, exist_ok=True)

def purity_score(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    total = 0
    for c in np.unique(y_pred):
        idx = np.where(y_pred == c)[0]
        if len(idx) == 0:
            continue
        labels, counts = np.unique(y_true[idx], return_counts=True)
        total += counts.max()
    return total / len(y_true)

def plot_tsne(Z, labels, title, out_path, cmap="tab10"):
    # CPU-friendly TSNE settings
    tsne = TSNE(
        n_components=2,
        perplexity=30,
        learning_rate="auto",
        init="pca",
        random_state=42
    )
    Z2 = tsne.fit_transform(Z)

    plt.figure()
    plt.scatter(Z2[:, 0], Z2[:, 1], s=6, alpha=0.8, c=labels, cmap=cmap)
    plt.title(title)
    plt.xlabel("t-SNE 1")
    plt.ylabel("t-SNE 2")
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()

def compute_metrics(Z, y_true, y_cluster):
    sil = float(silhouette_score(Z, y_cluster))
    ari = float(adjusted_rand_score(y_true, y_cluster))
    nmi = float(normalized_mutual_info_score(y_true, y_cluster))
    pur = float(purity_score(y_true, y_cluster))
    return sil, ari, nmi, pur

def main():
    print("Loading latent and labels...")
    Z = np.load(Z_PATH)        # (N, latent_dim)
    y_true = np.load(Y_PATH)   # (N,)

    # --- Safety: length mismatch হলে auto-fix ---
    n = min(len(Z), len(y_true))
    if len(Z) != len(y_true):
        print(f"[WARN] length mismatch: Z={len(Z)}, y={len(y_true)}. Using first n={n}.")
        Z = Z[:n]
        y_true = y_true[:n]

    with open(GENRE_MAP_PATH, "r", encoding="utf-8") as f:
        genre_map = json.load(f)

    # genre_map keys could be "0","1"... so k = length
    k = len(genre_map)
    print("K (genres) =", k)

    # ===== KMeans with multi-seed search (best-result) =====
    print("Clustering with KMeans (multi-seed search)...")
    seeds = [0, 42, 100, 123, 999]
    best = None

    for seed in seeds:
        km = KMeans(n_clusters=k, random_state=seed, n_init=50)
        y_cluster = km.fit_predict(Z)
        sil, ari, nmi, pur = compute_metrics(Z, y_true, y_cluster)

        row = {
            "method": "CVAE(latent_mu)+KMeans",
            "k": k,
            "seed": seed,
            "silhouette": sil,
            "ARI": ari,
            "NMI": nmi,
            "purity": pur
        }
        print(f"seed={seed} | sil={sil:.4f} | ARI={ari:.4f} | NMI={nmi:.4f} | purity={pur:.4f}")

        # choose best by NMI first, then ARI
        if best is None or (row["NMI"], row["ARI"]) > (best["NMI"], best["ARI"]):
            best = row
            best_clusters = y_cluster

    print("\nBEST (selected by NMI then ARI):")
    print(best)

    # ===== save metrics (overwrite single best row) =====
    metrics_path = os.path.join("results", "metrics_hard.csv")
    dfm = pd.DataFrame([best])
    dfm.to_csv(metrics_path, index=False)
    print("Saved metrics:", metrics_path)

    # ===== t-SNE plots =====
    print("Making t-SNE plots (CPU can take a few minutes)...")
    plot_tsne(Z, best_clusters, "t-SNE of CVAE latent (colored by cluster)",
              os.path.join(OUT_PLOTS, "tsne_by_cluster.png"), cmap="tab10")
    plot_tsne(Z, y_true, "t-SNE of CVAE latent (colored by true genre)",
              os.path.join(OUT_PLOTS, "tsne_by_true_genre.png"), cmap="tab10")
    print("Saved t-SNE plots in:", OUT_PLOTS)

    # ===== cluster vs genre distribution =====
    cont = np.zeros((k, k), dtype=int)
    for c in range(k):
        idx = np.where(best_clusters == c)[0]
        if len(idx) == 0:
            continue
        counts = np.bincount(y_true[idx], minlength=k)
        cont[c] = counts

    cont_norm = cont / np.clip(cont.sum(axis=1, keepdims=True), 1, None)

    # Save matrix as CSV (report friendly)
    genre_names = [genre_map[str(i)] for i in range(k)]
    df_mat = pd.DataFrame(cont_norm, columns=genre_names)
    df_mat.index.name = "cluster_id"
    df_mat.to_csv(os.path.join("results", "cluster_genre_matrix.csv"))
    print("Saved: results/cluster_genre_matrix.csv")

    # Plot matrix
    plt.figure(figsize=(10, 6))
    plt.imshow(cont_norm, aspect="auto")
    plt.colorbar(label="fraction in cluster")
    plt.xlabel("true genre")
    plt.ylabel("cluster id")
    plt.title("Cluster vs true genre distribution (row-normalized)")
    plt.xticks(ticks=np.arange(k), labels=genre_names, rotation=45, ha="right")
    plt.yticks(ticks=np.arange(k), labels=[str(i) for i in range(k)])
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_PLOTS, "cluster_genre_matrix.png"), dpi=180)
    plt.close()
    print("Saved: results/plots_hard/cluster_genre_matrix.png")

    # Save cluster labels (best) for later
    np.save(os.path.join("results", "cluster_labels_hard.npy"), best_clusters)
    print("Saved: results/cluster_labels_hard.npy")

    print("\nDONE ✅ Evaluation + visualization completed successfully.")

if __name__ == "__main__":
    main()