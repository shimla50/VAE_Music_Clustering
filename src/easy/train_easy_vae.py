import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

DATA_DIR = "data/processed/easy"
OUT_DIR = "results/easy"
os.makedirs(OUT_DIR, exist_ok=True)

class VAE(nn.Module):
    def __init__(self, input_dim, latent_dim=8):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 512)
        self.fc_mu = nn.Linear(512, latent_dim)
        self.fc_logvar = nn.Linear(512, latent_dim)
        self.fc2 = nn.Linear(latent_dim, 512)
        self.fc3 = nn.Linear(512, input_dim)

    def encode(self, x):
        h = torch.relu(self.fc1(x))
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h = torch.relu(self.fc2(z))
        return self.fc3(h)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

def loss_fn(recon, x, mu, logvar):
    recon_loss = nn.functional.mse_loss(recon, x, reduction="mean")
    kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + kl

def main():
    X = np.load(os.path.join(DATA_DIR, "X_easy.npy"))
    X = torch.tensor(X, dtype=torch.float32)

    loader = DataLoader(TensorDataset(X), batch_size=64, shuffle=True)

    model = VAE(X.shape[1])
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(1, 31):
        total = 0
        for (xb,) in loader:
            recon, mu, logvar = model(xb)
            loss = loss_fn(recon, xb, mu, logvar)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item()

        if epoch % 5 == 0:
            print(f"Epoch {epoch} | loss={total/len(loader):.4f}")

    with torch.no_grad():
        mu, _ = model.encode(X)
        Z = mu.numpy()

    np.save(os.path.join(OUT_DIR, "latent_easy.npy"), Z)
    print("Saved latent:", Z.shape)

if __name__ == "__main__":
    main()
