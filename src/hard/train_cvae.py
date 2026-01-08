print("CVAE training started", flush=True)

import os
import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import pandas as pd

# =========================
# CLEAN OUTPUT PATHS
# =========================
RESULTS_DIR = os.path.join("results", "hard")
PLOTS_DIR = os.path.join(RESULTS_DIR, "plots")
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)

# ===== paths =====
X_PATH = os.path.join("data", "hard", "features", "X_fusion.npy")
Y_PATH = os.path.join("data", "hard", "features", "y_genre.npy")
GENRE_MAP_PATH = os.path.join("data", "hard", "features", "genre_map.json")

RANDOM_SEED = 42
torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# ===== CPU-only =====
DEVICE = torch.device("cpu")

# ===== training settings (CPU friendly) =====
BATCH_SIZE = 64
EPOCHS = 30
LR = 1e-3
LATENT_DIM = 32
HIDDEN1 = 512
HIDDEN2 = 256
BETA = 1.0

def one_hot(y: np.ndarray, num_classes: int) -> np.ndarray:
    out = np.zeros((len(y), num_classes), dtype=np.float32)
    out[np.arange(len(y)), y] = 1.0
    return out

class FusionDataset(Dataset):
    def __init__(self, X, y, y_oh):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)
        self.y_oh = torch.tensor(y_oh, dtype=torch.float32)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx], self.y_oh[idx]

class CVAE(nn.Module):
    def __init__(self, x_dim, y_dim, latent_dim=32, h1=512, h2=256):
        super().__init__()
        self.x_dim = x_dim
        self.y_dim = y_dim
        self.latent_dim = latent_dim

        self.enc = nn.Sequential(
            nn.Linear(x_dim + y_dim, h1),
            nn.ReLU(),
            nn.Linear(h1, h2),
            nn.ReLU(),
        )
        self.mu = nn.Linear(h2, latent_dim)
        self.logvar = nn.Linear(h2, latent_dim)

        self.dec = nn.Sequential(
            nn.Linear(latent_dim + y_dim, h2),
            nn.ReLU(),
            nn.Linear(h2, h1),
            nn.ReLU(),
            nn.Linear(h1, x_dim),
        )

    def encode(self, x, y_oh):
        h = self.enc(torch.cat([x, y_oh], dim=1))
        return self.mu(h), self.logvar(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z, y_oh):
        return self.dec(torch.cat([z, y_oh], dim=1))

    def forward(self, x, y_oh):
        mu, logvar = self.encode(x, y_oh)
        z = self.reparameterize(mu, logvar)
        x_hat = self.decode(z, y_oh)
        return x_hat, mu, logvar, z

def loss_fn(x, x_hat, mu, logvar, beta=1.0):
    recon = torch.mean((x_hat - x) ** 2)
    kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - torch.exp(logvar))
    return recon + beta * kl, recon.detach(), kl.detach()

def main():
    print("Loading features...")
    X = np.load(X_PATH)
    y = np.load(Y_PATH)
    with open(GENRE_MAP_PATH, "r", encoding="utf-8") as f:
        genre_map = json.load(f)

    num_classes = len(genre_map)
    y_oh = one_hot(y, num_classes)

    idx = np.arange(len(X))
    np.random.shuffle(idx)
    split = int(0.9 * len(X))
    tr_idx, va_idx = idx[:split], idx[split:]

    X_tr, y_tr, yoh_tr = X[tr_idx], y[tr_idx], y_oh[tr_idx]
    X_va, y_va, yoh_va = X[va_idx], y[va_idx], y_oh[va_idx]

    train_ds = FusionDataset(X_tr, y_tr, yoh_tr)
    val_ds = FusionDataset(X_va, y_va, yoh_va)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

    model = CVAE(x_dim=X.shape[1], y_dim=num_classes, latent_dim=LATENT_DIM, h1=HIDDEN1, h2=HIDDEN2).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=LR)

    train_losses, val_losses = [], []

    print("Training CVAE...")
    for epoch in range(1, EPOCHS + 1):
        model.train()
        tl = []
        for xb, _, yb_oh in train_loader:
            xb = xb.to(DEVICE)
            yb_oh = yb_oh.to(DEVICE)

            x_hat, mu, logvar, _ = model(xb, yb_oh)
            loss, _, _ = loss_fn(xb, x_hat, mu, logvar, beta=BETA)

            opt.zero_grad()
            loss.backward()
            opt.step()
            tl.append(loss.item())

        model.eval()
        vl = []
        with torch.no_grad():
            for xb, _, yb_oh in val_loader:
                xb = xb.to(DEVICE)
                yb_oh = yb_oh.to(DEVICE)
                x_hat, mu, logvar, _ = model(xb, yb_oh)
                loss, _, _ = loss_fn(xb, x_hat, mu, logvar, beta=BETA)
                vl.append(loss.item())

        train_loss = float(np.mean(tl))
        val_loss = float(np.mean(vl))
        train_losses.append(train_loss)
        val_losses.append(val_loss)

        print(f"Epoch {epoch:02d}/{EPOCHS} | train={train_loss:.4f} | val={val_loss:.4f}")

    # Save model
    model_path = os.path.join(RESULTS_DIR, "cvae.pth")
    torch.save(model.state_dict(), model_path)
    print("Saved model:", model_path)

    # Extract latent means
    model.eval()
    with torch.no_grad():
        X_t = torch.tensor(X, dtype=torch.float32).to(DEVICE)
        yoh_t = torch.tensor(y_oh, dtype=torch.float32).to(DEVICE)
        mu, _ = model.encode(X_t, yoh_t)
        Z = mu.cpu().numpy().astype(np.float32)

    z_path = os.path.join(RESULTS_DIR, "latent_hard.npy")
    np.save(z_path, Z)
    print("Saved latent:", z_path, Z.shape)

    # Reconstruction
    with torch.no_grad():
        x_hat = model.decode(torch.tensor(Z, dtype=torch.float32).to(DEVICE), yoh_t).cpu().numpy().astype(np.float32)

    mse = np.mean((x_hat - X.astype(np.float32)) ** 2, axis=1)

    df_err = pd.DataFrame({"recon_mse": mse})
    err_path = os.path.join(RESULTS_DIR, "recon_errors.csv")
    df_err.to_csv(err_path, index=False)
    print("Saved recon errors:", err_path)

    top_idx = np.argsort(mse)[:10]
    bottom_idx = np.argsort(mse)[-10:][::-1]
    df_examples = pd.DataFrame({
        "type": ["best"] * 10 + ["worst"] * 10,
        "index": np.concatenate([top_idx, bottom_idx]),
        "recon_mse": np.concatenate([mse[top_idx], mse[bottom_idx]])
    })
    ex_path = os.path.join(RESULTS_DIR, "recon_examples_top_bottom.csv")
    df_examples.to_csv(ex_path, index=False)
    print("Saved recon example indices:", ex_path)

    plt.figure()
    plt.hist(mse, bins=50)
    plt.title("Reconstruction error (MSE) distribution")
    plt.xlabel("MSE")
    plt.ylabel("Count")
    plt.tight_layout()
    p1 = os.path.join(PLOTS_DIR, "recon_error_hist.png")
    plt.savefig(p1, dpi=150)
    plt.close()
    print("Saved plot:", p1)

    plt.figure()
    plt.plot(train_losses, label="train")
    plt.plot(val_losses, label="val")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.legend()
    plt.title("CVAE training loss")
    plt.tight_layout()
    p2 = os.path.join(PLOTS_DIR, "cvae_loss.png")
    plt.savefig(p2, dpi=150)
    plt.close()
    print("Saved plot:", p2)

if __name__ == "__main__":
    main()
