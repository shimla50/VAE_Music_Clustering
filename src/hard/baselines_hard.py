import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, adjusted_rand_score, normalized_mutual_info_score

RESULTS_DIR = os.path.join("results", "hard")
PLOTS_DIR = os.path.join(RESULTS_DIR, "plots")
os.makedirs(PLOTS_DIR, exist_ok=True)

# ---------- Paths ----------
X_FUSION_PATH = os.path.join("data", "hard", "features", "X_fusion.npy")
X_AUDIO_PATH = os.path.join("data", "hard", "features", "X_audio.npy")
Y_PATH = os.path.join("data", "hard", "features", "y_genre.npy")
GENRE_MAP_PATH = os.path.join("data", "hard", "features", "genre_map.json")

# ---------- Metrics ----------
def purity_score(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    total = 0
    for c in np.unique(y_pred):
        idx = np.where(y_pred == c)[0]
        if len(idx) == 0:
            continue
        _, counts = np.unique(y_true[idx], return_counts=True)
        total += counts.max()
    return total / len(y_true)

def eval_kmeans(X, y_true, k, tag, seed=42):
    km = KMeans(n_clusters=k, random_state=seed, n_init=50)
    y_pred = km.fit_predict(X)

    sil = float(silhouette_score(X, y_pred))
    ari = float(adjusted_rand_score(y_true, y_pred))
    nmi = float(normalized_mutual_info_score(y_true, y_pred))
    pur = float(purity_score(y_true, y_pred))

    return {
        "method": tag,
        "k": k,
        "silhouette": sil,
        "ARI": ari,
        "NMI": nmi,
        "purity": pur
    }, y_pred

# ---------- Autoencoder baseline ----------
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

class AE(nn.Module):
    def __init__(self, in_dim, latent_dim=8):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(in_dim, 64),
            nn.ReLU(),
            nn.Linear(64, latent_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ReLU(),
            nn.Linear(64, in_dim),
        )

    def forward(self, x):
        z = self.encoder(x)
        xhat = self.decoder(z)
        return xhat, z

def train_ae_get_latent(X, latent_dim=8, epochs=50, batch_size=256, lr=1e-3):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    X_t = torch.tensor(X, dtype=torch.float32)
    dl = DataLoader(TensorDataset(X_t), batch_size=batch_size, shuffle=True)

    model = AE(X.shape[1], latent_dim=latent_dim).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    model.train()
    for ep in range(1, epochs + 1):
        total = 0.0
        for (xb,) in dl:
            xb = xb.to(device)
            xhat, _ = model(xb)
            loss = loss_fn(xhat, xb)

            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item() * xb.size(0)

        if ep % 10 == 0 or ep == 1:
            print(f"AE Epoch {ep:02d}/{epochs} | loss={total/len(X):.6f}")

    model.eval()
    with torch.no_grad():
        Z = model.encoder(X_t.to(device)).cpu().numpy()
    return Z

def main():
    print("Loading labels and features...")
    y_true = np.load(Y_PATH)

    with open(GENRE_MAP_PATH, "r", encoding="utf-8") as f:
        genre_map = json.load(f)
    k = len(genre_map)

    X_audio = np.load(X_AUDIO_PATH).astype(np.float32)
    X_fusion = np.load(X_FUSION_PATH).astype(np.float32)

    n = min(len(y_true), len(X_audio), len(X_fusion))
    if not (len(y_true) == len(X_audio) == len(X_fusion)):
        print(f"[WARN] length mismatch. Using first n={n}.")
        y_true = y_true[:n]
        X_audio = X_audio[:n]
        X_fusion = X_fusion[:n]

    rows = []

    print("\nBaseline 0: Direct X_audio + KMeans")
    r_audio, _ = eval_kmeans(X_audio, y_true, k, "Direct(X_audio)+KMeans")
    rows.append(r_audio)

    print("\nBaseline 1: AE(X_audio) + KMeans")
    Z_ae = train_ae_get_latent(X_audio, latent_dim=8, epochs=50, batch_size=256, lr=1e-3)
    r_ae, _ = eval_kmeans(Z_ae, y_true, k, "AE(audio_latent8)+KMeans")
    rows.append(r_ae)

    print("\nBaseline 2: Direct X_fusion + KMeans")
    r_fusion, _ = eval_kmeans(X_fusion, y_true, k, "Direct(X_fusion)+KMeans")
    rows.append(r_fusion)

    print("\nBaseline 3: PCA(50) + KMeans")
    pca50 = PCA(n_components=50, random_state=42)
    X50 = pca50.fit_transform(X_fusion)
    r_pca50, _ = eval_kmeans(X50, y_true, k, "PCA(50)+KMeans")
    rows.append(r_pca50)

    print("\nBaseline 4: PCA(20) + KMeans")
    pca20 = PCA(n_components=20, random_state=42)
    X20 = pca20.fit_transform(X_fusion)
    r_pca20, _ = eval_kmeans(X20, y_true, k, "PCA(20)+KMeans")
    rows.append(r_pca20)

    print("\nBaseline 5: PCA(10) + KMeans")
    pca10 = PCA(n_components=10, random_state=42)
    X10 = pca10.fit_transform(X_fusion)
    r_pca10, _ = eval_kmeans(X10, y_true, k, "PCA(10)+KMeans")
    rows.append(r_pca10)

    print("\nBaseline 6: PCA(2) + KMeans + plot")
    pca2 = PCA(n_components=2, random_state=42)
    X2 = pca2.fit_transform(X_fusion)
    r_pca2, y_pred2 = eval_kmeans(X2, y_true, k, "PCA(2)+KMeans")
    rows.append(r_pca2)

    plt.figure()
    plt.scatter(X2[:, 0], X2[:, 1], s=6, alpha=0.8, c=y_pred2, cmap="tab10")
    plt.title("PCA(2) baseline (colored by KMeans clusters)")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.tight_layout()
    p = os.path.join(PLOTS_DIR, "pca2_baseline.png")
    plt.savefig(p, dpi=180)
    plt.close()
    print("Saved plot:", p)

    out_csv = os.path.join(RESULTS_DIR, "baselines_hard.csv")
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    print("Saved:", out_csv)

    print("\nDONE ✅ Baselines completed successfully.")

if __name__ == "__main__":
    main()
